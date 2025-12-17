// scratch/urllc-gen.cc
// 1 gNB + 1 UE (5G-LENA NR) + variable-size UDP + CSV with:
// inter-arrival (us), inter-arrival jitter (us), one-way latency (us),
// tx_time, tx_payload_bytes, seq/gap, throughput over a 50 ms sliding window.

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/nr-module.h"
#include "ns3/timestamp-tag.h"

#include <fstream>
#include <iomanip>
#include <vector>
#include <functional>
#include <algorithm>
#include <deque>

using namespace ns3;

// -------------------- Custom packet tags: SeqTag + TxSizeTag --------------------
class SeqTag : public Tag
{
public:
  SeqTag() : m_seq(0) {}
  explicit SeqTag(uint64_t s) : m_seq(s) {}

  static TypeId GetTypeId()
  {
    static TypeId tid = TypeId("ns3::SeqTag").SetParent<Tag>().AddConstructor<SeqTag>();
    return tid;
  }

  TypeId GetInstanceTypeId() const override { return GetTypeId(); }
  uint32_t GetSerializedSize() const override { return 8; }
  void Serialize(TagBuffer i) const override { i.WriteU64(m_seq); }
  void Deserialize(TagBuffer i) override { m_seq = i.ReadU64(); }
  void Print(std::ostream& os) const override { os << m_seq; }

  void SetSeq(uint64_t s) { m_seq = s; }
  uint64_t GetSeq() const { return m_seq; }

private:
  uint64_t m_seq;
};

class TxSizeTag : public Tag
{
public:
  TxSizeTag() : m_size(0) {}
  explicit TxSizeTag(uint32_t s) : m_size(s) {}

  static TypeId GetTypeId()
  {
    static TypeId tid = TypeId("ns3::TxSizeTag").SetParent<Tag>().AddConstructor<TxSizeTag>();
    return tid;
  }

  TypeId GetInstanceTypeId() const override { return GetTypeId(); }
  uint32_t GetSerializedSize() const override { return 4; }
  void Serialize(TagBuffer i) const override { i.WriteU32(m_size); }
  void Deserialize(TagBuffer i) override { m_size = i.ReadU32(); }
  void Print(std::ostream& os) const override { os << m_size; }

  void SetSize(uint32_t s) { m_size = s; }
  uint32_t GetSize() const { return m_size; }

private:
  uint32_t m_size;
};

// -------------------- Variable-size UDP sender --------------------
class VariableSizeUdpApp : public Application
{
public:
  void Setup(Address peer, uint32_t minBytes, uint32_t maxBytes, Time interval)
  {
    m_peer = peer;
    m_min = std::min(minBytes, maxBytes);
    m_max = std::max(minBytes, maxBytes);
    m_interval = interval;
    m_rng = CreateObject<UniformRandomVariable>();
  }

private:
  void StartApplication() override
  {
    m_running = true;
    if (!m_socket)
    {
      m_socket = Socket::CreateSocket(GetNode(), UdpSocketFactory::GetTypeId());
      m_socket->Connect(m_peer);
    }
    SendOnce();
  }

  void StopApplication() override
  {
    m_running = false;
    if (m_ev.IsPending())
      Simulator::Cancel(m_ev);
    if (m_socket)
    {
      m_socket->Close();
      m_socket = nullptr;
    }
  }

  void SendOnce()
  {
    if (!m_running) return;

    uint32_t sz = m_rng->GetInteger(m_min, m_max);
    Ptr<Packet> p = Create<Packet>(sz);

    TimestampTag ts;
    ts.SetTimestamp(Simulator::Now());
    p->AddPacketTag(ts);

    SeqTag st;
    st.SetSeq(m_seq++);
    p->AddPacketTag(st);

    TxSizeTag tt;
    tt.SetSize(sz);
    p->AddPacketTag(tt);

    m_socket->Send(p);
    m_ev = Simulator::Schedule(m_interval, &VariableSizeUdpApp::SendOnce, this);
  }

private:
  Address m_peer;
  Ptr<Socket> m_socket;
  EventId m_ev;
  bool m_running{false};

  uint32_t m_min{32}, m_max{256};
  Time m_interval{MilliSeconds(1)};
  Ptr<UniformRandomVariable> m_rng;

  uint64_t m_seq{0};
};

// -------------------- CSV logging (PacketSink Rx trace) --------------------
static std::ofstream g_out;

static bool g_hasLastRx = false;
static Time g_lastRx;
static uint64_t g_idx = 0;

static bool g_hasLastSeq = false;
static uint64_t g_lastSeq = 0;

// inter-arrival jitter state (microseconds)
static bool g_hasLastIat = false;
static double g_lastIatUs = 0.0;

// 50 ms sliding window throughput
static const Time g_thrWindow = MilliSeconds(50);
static std::deque<std::pair<Time, uint32_t>> g_win; // (rx_time, bytes)
static uint64_t g_winBytes = 0;

static void
LogRx(std::ofstream* out, Ptr<const Packet> p, const Address& from)
{
  Time now = Simulator::Now();

  // inter-arrival in microseconds
  double iatUs = -1.0;
  if (g_hasLastRx)
    iatUs = (now - g_lastRx).GetMicroSeconds();
  g_lastRx = now;
  g_hasLastRx = true;

  // tx_time_s + latency in microseconds
  double txTimeS = -1.0;
  double latencyUs = -1.0;
  TimestampTag ts;
  if (p->PeekPacketTag(ts))
  {
    Time ttx = ts.GetTimestamp();
    txTimeS = ttx.GetSeconds();
    latencyUs = (now - ttx).GetMicroSeconds();
  }

  // inter-arrival jitter in microseconds
  double jitterUs = 0.0;
  if (iatUs >= 0.0)
  {
    if (g_hasLastIat)
      jitterUs = std::abs(iatUs - g_lastIatUs);
    else
      jitterUs = 0.0;

    g_lastIatUs = iatUs;
    g_hasLastIat = true;
  }

  // seq + gap
  uint64_t seq = 0;
  int64_t gap = -1;
  SeqTag st;
  if (p->PeekPacketTag(st))
  {
    seq = st.GetSeq();
    if (g_hasLastSeq)
    {
      if (seq > g_lastSeq)
        gap = static_cast<int64_t>(seq - g_lastSeq - 1);
      else
        gap = 0;
    }
    else
    {
      gap = 0;
    }
    g_lastSeq = seq;
    g_hasLastSeq = true;
  }

  // tx payload bytes
  uint32_t txPayloadBytes = 0;
  TxSizeTag tt;
  if (p->PeekPacketTag(tt))
    txPayloadBytes = tt.GetSize();
  else
    txPayloadBytes = p->GetSize();

  // rx payload bytes
  uint32_t rxPayloadBytes = p->GetSize();

  // 50 ms window throughput (Mbps) based on received bytes
  g_win.emplace_back(now, rxPayloadBytes);
  g_winBytes += rxPayloadBytes;

  Time cutoff = now - g_thrWindow;
  while (!g_win.empty() && g_win.front().first < cutoff)
  {
    g_winBytes -= g_win.front().second;
    g_win.pop_front();
  }

  double denomS = g_thrWindow.GetSeconds();
  if (!g_win.empty())
  {
    Time span = now - g_win.front().first;
    double spanS = span.GetSeconds();
    if (spanS > 1e-9 && spanS < denomS) denomS = spanS;
  }

  double thrMbps = 0.0;
  if (denomS > 1e-9)
    thrMbps = (g_winBytes * 8.0 / denomS) / 1e6;

  InetSocketAddress src = InetSocketAddress::ConvertFrom(from);

  (*out) << std::fixed << std::setprecision(6)
         << now.GetSeconds() << ","
         << g_idx++ << ","
         << rxPayloadBytes << ","
         << txTimeS << ","
         << txPayloadBytes << ","
         << seq << ","
         << gap << ","
         << std::setprecision(0) << iatUs << ","
         << std::setprecision(0) << latencyUs << ","
         << std::setprecision(0) << jitterUs << ","
         << std::setprecision(6) << thrMbps << ","
         << src.GetIpv4() << ","
         << src.GetPort()
         << "\n";
}

int main(int argc, char* argv[])
{
  double simTime = 5.0;
  double appStart = 1.0;

  double frequency = 3.5e9;
  double bandwidth = 20e6;
  uint8_t numerology = 2;
  double gnbTxPowerDbm = 30.0;

  uint32_t minPkt = 40;
  uint32_t maxPkt = 600;
  double intervalMs = 1.0;
  uint16_t dstPort = 5000;

  std::string csv = "out.csv";

  CommandLine cmd;
  cmd.AddValue("simTime", "Simulation time (s)", simTime);
  cmd.AddValue("appStart", "App start time (s)", appStart);
  cmd.AddValue("frequency", "Carrier frequency (Hz)", frequency);
  cmd.AddValue("bandwidth", "Bandwidth (Hz)", bandwidth);
  cmd.AddValue("numerology", "NR numerology mu", numerology);
  cmd.AddValue("gnbTxPowerDbm", "gNB Tx power (dBm)", gnbTxPowerDbm);
  cmd.AddValue("minPkt", "Min UDP payload bytes", minPkt);
  cmd.AddValue("maxPkt", "Max UDP payload bytes", maxPkt);
  cmd.AddValue("intervalMs", "Send interval (ms)", intervalMs);
  cmd.AddValue("csv", "Output CSV filename", csv);
  cmd.Parse(argc, argv);

  Config::SetDefault("ns3::NrRlcUm::MaxTxBufferSize", UintegerValue(999999999));

  NodeContainer gnbNodes, ueNodes;
  gnbNodes.Create(1);
  ueNodes.Create(1);

  MobilityHelper mob;
  mob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
  mob.Install(gnbNodes);
  mob.Install(ueNodes);

  gnbNodes.Get(0)->GetObject<MobilityModel>()->SetPosition(Vector(0.0, 0.0, 10.0));
  ueNodes.Get(0)->GetObject<MobilityModel>()->SetPosition(Vector(10.0, 0.0, 1.5));

  Ptr<NrPointToPointEpcHelper> epcHelper = CreateObject<NrPointToPointEpcHelper>();
  Ptr<IdealBeamformingHelper> bfHelper = CreateObject<IdealBeamformingHelper>();
  Ptr<NrHelper> nrHelper = CreateObject<NrHelper>();
  nrHelper->SetEpcHelper(epcHelper);
  nrHelper->SetBeamformingHelper(bfHelper);

  CcBwpCreator ccBwpCreator;
  CcBwpCreator::SimpleOperationBandConf bandConf(frequency, bandwidth, 1);
  OperationBandInfo band = ccBwpCreator.CreateOperationBandContiguousCc(bandConf);

  Ptr<NrChannelHelper> ch = CreateObject<NrChannelHelper>();
  ch->ConfigureFactories("UMi", "Default", "ThreeGpp");
  ch->SetPathlossAttribute("ShadowingEnabled", BooleanValue(false));
  ch->AssignChannelsToBands({band});

  BandwidthPartInfoPtrVector bwps = CcBwpCreator::GetAllBwps({band});
  std::vector<std::reference_wrapper<BandwidthPartInfoPtr>> bwpRefs;
  bwpRefs.reserve(bwps.size());
  for (auto& b : bwps) bwpRefs.emplace_back(b);

  NetDeviceContainer gnbDevs = nrHelper->InstallGnbDevice(gnbNodes, bwpRefs);
  NetDeviceContainer ueDevs  = nrHelper->InstallUeDevice(ueNodes,  bwpRefs);

  NrHelper::GetGnbPhy(gnbDevs.Get(0), 0)->SetAttribute("Numerology", UintegerValue(numerology));
  NrHelper::GetGnbPhy(gnbDevs.Get(0), 0)->SetAttribute("TxPower", DoubleValue(gnbTxPowerDbm));

  InternetStackHelper internet;
  internet.Install(ueNodes);

  NodeContainer remoteHostContainer;
  remoteHostContainer.Create(1);
  Ptr<Node> remoteHost = remoteHostContainer.Get(0);
  internet.Install(remoteHostContainer);

  Ptr<Node> pgw = epcHelper->GetPgwNode();

  PointToPointHelper p2p;
  p2p.SetDeviceAttribute("DataRate", DataRateValue(DataRate("10Gb/s")));
  p2p.SetChannelAttribute("Delay", TimeValue(MilliSeconds(1)));
  NetDeviceContainer internetDevs = p2p.Install(pgw, remoteHost);

  Ipv4AddressHelper ipv4h;
  ipv4h.SetBase("1.0.0.0", "255.0.0.0");
  Ipv4InterfaceContainer internetIfaces = ipv4h.Assign(internetDevs);

  Ipv4StaticRoutingHelper routing;

  Ptr<Ipv4StaticRouting> rhRoute = routing.GetStaticRouting(remoteHost->GetObject<Ipv4>());
  rhRoute->AddNetworkRouteTo(Ipv4Address("7.0.0.0"), Ipv4Mask("255.0.0.0"), 1);

  Ipv4InterfaceContainer ueIfaces = epcHelper->AssignUeIpv4Address(NetDeviceContainer(ueDevs));
  Ptr<Ipv4StaticRouting> ueRoute = routing.GetStaticRouting(ueNodes.Get(0)->GetObject<Ipv4>());
  ueRoute->SetDefaultRoute(epcHelper->GetUeDefaultGatewayAddress(), 1);

  nrHelper->AttachToClosestGnb(ueDevs, gnbDevs);

  g_out.open(csv, std::ios::out | std::ios::trunc);
  g_out << "rx_time_s,idx,rx_payload_bytes,tx_time_s,tx_payload_bytes,seq,gap,"
           "inter_arrival_us,latency_us,jitter_us,throughput_mbps_50ms,src_ip,src_port\n";

  PacketSinkHelper sinkHelper("ns3::UdpSocketFactory",
                              InetSocketAddress(Ipv4Address::GetAny(), dstPort));
  ApplicationContainer sinkApps = sinkHelper.Install(ueNodes.Get(0));
  sinkApps.Start(Seconds(appStart));
  sinkApps.Stop(Seconds(simTime));

  Ptr<PacketSink> sink = DynamicCast<PacketSink>(sinkApps.Get(0));
  sink->TraceConnectWithoutContext("Rx", MakeBoundCallback(&LogRx, &g_out));

  Address peer = InetSocketAddress(ueIfaces.GetAddress(0), dstPort);
  Ptr<VariableSizeUdpApp> sender = CreateObject<VariableSizeUdpApp>();
  sender->Setup(peer, minPkt, maxPkt, MilliSeconds(intervalMs));
  remoteHost->AddApplication(sender);
  sender->SetStartTime(Seconds(appStart));
  sender->SetStopTime(Seconds(simTime));

  Simulator::Stop(Seconds(simTime));
  Simulator::Run();
  Simulator::Destroy();

  g_out.close();
  return 0;
}
