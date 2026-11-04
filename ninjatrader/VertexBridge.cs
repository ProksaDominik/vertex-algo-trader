// VertexBridge -- NinjaTrader 8 add-on for the Vertex Algo Bot.
//
// Lets the Python bot (src/metatraderbot/nt_engine.py) trade the account
// NinjaTrader is connected to -- Lucid Trading over Rithmic or Tradovate, or
// NinjaTrader's own Sim101 for testing. It listens on 127.0.0.1 ONLY (port
// 45973, or the NT_BRIDGE_PORT environment variable) and answers one request
// per connection: a tab-separated line in, one JSON line out.
//
// Install: run ninjatrader\Install-VertexBridge.ps1 (or copy this file to
// Documents\NinjaTrader 8\bin\Custom\AddOns\), then in NinjaTrader open
// New > NinjaScript Editor and press F5 to compile. The Output window
// (New > NinjaScript Output) prints "VertexBridge listening on ..." once it runs.
//
// What it does beyond passing orders through:
//   * Every order it places is named VX|<magic>|<ticket>|<role> (E entry,
//     S stop, T target, X exit), so a ticket is recognisable after a restart
//     and an entry retried by the bot is never placed twice.
//   * The moment an entry fills it places the stop-loss and take-profit as an
//     OCO pair at the fill price +/- the distances the bot asked for.
//   * Fail-closed: a rejected or vanished stop gets the position closed at
//     market; protection larger than the position is shrunk so it can never
//     reverse it; a position closed outside the bot (Lucid's 4:45 PM
//     auto-close, a manual flatten) is detected and its leftover orders are
//     cancelled so they cannot open a new position later.
//   * One open bot ticket per instrument per account: futures accounts net
//     positions per instrument, so two tickets would share one position.
//
// Written in C# 5 style (no ?. or $"") so it compiles on every NinjaTrader 8
// release. APIs that differ between releases are reached through reflection.

#region Using declarations
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Reflection;
using System.Text;
using System.Threading;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
#endregion

namespace NinjaTrader.NinjaScript.AddOns
{
	public class VertexBridge : AddOnBase
	{
		// NOT 36973: that is NinjaTrader's own ATI port, already bound on every
		// install -- a client reaching it gets no answer and simply hangs.
		private const int DefaultPort = 45973;
		private const string Prefix = "VX|";
		// 1.1: never acts on a position the account does not hold (the 1.0 audit
		// once opened a short "protecting" a trade that was already closed), and
		// exits are uniquely named. nt_engine refuses to trade through anything older.
		// 1.3: "flow" -- order flow (buying vs selling volume per minute) for research.
		private const string BridgeVersion = "1.4";
		private static readonly DateTime UnixEpoch = new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);

		private static readonly object ServerLock = new object();
		private static VertexBridge activeInstance;
		private static TimeZoneInfo ntZone;

		private readonly object sync = new object();
		private readonly Dictionary<long, Ticket> tickets = new Dictionary<long, Ticket>();
		private readonly Dictionary<string, Quote> quotes = new Dictionary<string, Quote>();
		private readonly Dictionary<string, MarketData> feeds = new Dictionary<string, MarketData>();
		private readonly Dictionary<MarketData, string> feedKeys = new Dictionary<MarketData, string>();
		private readonly Dictionary<string, DateTime> feedStarted = new Dictionary<string, DateTime>();   // 1.4: last (re)subscribe
		private readonly Dictionary<string, DateTime> fillTimes = new Dictionary<string, DateTime>();   // order name -> last fill
		private readonly List<Account> hooked = new List<Account>();

		private TcpListener listener;
		private Thread acceptThread;
		private System.Threading.Timer auditTimer;
		private volatile bool running;
		private int auditBusy;

		private class Ticket
		{
			public long Id;
			public int Magic;
			public string AccountName;
			public string InstrumentName;
			public bool IsLong;
			public int Qty;
			public double SlDist;
			public double TpDist;
			public Order Entry;
			public Order Stop;
			public Order Target;
			// Every S, T and X order ever placed, keyed by its (unique) name and
			// holding the NEWEST instance. NinjaTrader passes a fresh Order object
			// with each update, so tracking by reference counted every fill twice.
			public Dictionary<string, Order> Exits = new Dictionary<string, Order>();
			public int ExitSeq;
			public bool BracketSent;
			public int ProtectedQty;
			public string BracketState = "pending";             // pending | working | failed | none
			public string Error = "";
			public string CloseReason = "";
			public DateTime OpenedUtc;
			public DateTime? ClosedUtc;
			public bool Closing;
			public bool Finalized;
			public bool Rebuilt;
			public int ExternalExitQty;
			public double ExternalExitNotional;
			public int OcoSeq;
			public int Suspect;
			public int Mismatch;
			public int SizeDrift;
		}

		private class Quote
		{
			public double Bid = double.NaN;
			public double Ask = double.NaN;
			public double Last = double.NaN;
			public DateTime Utc = DateTime.MinValue;

			public bool HasBidAsk
			{
				get { return !double.IsNaN(Bid) && !double.IsNaN(Ask) && Bid > 0 && Ask > 0; }
			}
		}

		protected override void OnStateChange()
		{
			if (State == State.SetDefaults)
			{
				Name = "VertexBridge";
				Description = "Local bridge for the Vertex Algo Bot (127.0.0.1 only).";
			}
			else if (State == State.Configure || State == State.Active)
			{
				// Add-ons pass through both; start on whichever comes first.
				if (!running)
					StartBridge();
			}
			else if (State == State.Terminated)
			{
				StopBridge();
			}
		}

		// ------------------------------------------------------------------
		// server
		// ------------------------------------------------------------------

		private void StartBridge()
		{
			lock (ServerLock)
			{
				if (activeInstance != null && activeInstance != this)
					activeInstance.StopBridge();
				activeInstance = this;
			}
			running = true;
			acceptThread = new Thread(AcceptLoop);
			acceptThread.IsBackground = true;
			acceptThread.Name = "VertexBridge";
			acceptThread.Start();
			auditTimer = new System.Threading.Timer(AuditTick, null, 2000, 2000);
		}

		private void StopBridge()
		{
			running = false;
			try { if (listener != null) listener.Stop(); } catch { }
			try { if (auditTimer != null) auditTimer.Dispose(); } catch { }
			lock (sync)
			{
				foreach (Account a in hooked)
				{
					try { a.OrderUpdate -= OnOrderUpdate; } catch { }
				}
				hooked.Clear();
				foreach (MarketData md in feeds.Values)
				{
					try { md.Update -= OnMarketData; } catch { }
				}
				feeds.Clear();
				feedKeys.Clear();
				feedStarted.Clear();
				quotes.Clear();
			}
			lock (ServerLock)
			{
				if (activeInstance == this)
					activeInstance = null;
			}
		}

		private static int Port()
		{
			string s = Environment.GetEnvironmentVariable("NT_BRIDGE_PORT");
			int p;
			if (!string.IsNullOrEmpty(s) && int.TryParse(s, out p) && p > 0 && p < 65536)
				return p;
			return DefaultPort;
		}

		private void AcceptLoop()
		{
			int port = Port();
			while (running)
			{
				try
				{
					listener = new TcpListener(IPAddress.Loopback, port);
					listener.Start();
					Print("VertexBridge " + BridgeVersion + " listening on 127.0.0.1:" + port);
					while (running)
					{
						TcpClient client = listener.AcceptTcpClient();
						ThreadPool.QueueUserWorkItem(delegate { Serve(client); });
					}
				}
				catch (Exception ex)
				{
					if (!running)
						break;
					// Usually the port is still held by the instance a recompile is replacing.
					Print("VertexBridge: listener error (" + ex.Message + "), retrying in 3s");
					try { if (listener != null) listener.Stop(); } catch { }
					Thread.Sleep(3000);
				}
			}
		}

		private void Serve(TcpClient client)
		{
			using (client)
			{
				try
				{
					client.ReceiveTimeout = 10000;
					client.SendTimeout = 10000;
					NetworkStream stream = client.GetStream();
					StreamReader reader = new StreamReader(stream, new UTF8Encoding(false));
					string line = reader.ReadLine();
					if (line == null)
						return;
					string reply;
					try
					{
						reply = "{\"ok\":true,\"data\":" + Dispatch(line) + "}";
					}
					catch (Exception ex)
					{
						reply = "{\"ok\":false,\"error\":" + J.Str(ex.Message) + "}";
					}
					byte[] bytes = new UTF8Encoding(false).GetBytes(reply + "\n");
					stream.Write(bytes, 0, bytes.Length);
					stream.Flush();
				}
				catch (Exception ex)
				{
					Print("VertexBridge: client error " + ex.Message);
				}
			}
		}

		private string Dispatch(string line)
		{
			string[] parts = line.TrimEnd('\r').Split('\t');
			Dictionary<string, string> a = new Dictionary<string, string>();
			for (int i = 1; i < parts.Length; i++)
			{
				int eq = parts[i].IndexOf('=');
				if (eq > 0)
					a[parts[i].Substring(0, eq)] = parts[i].Substring(eq + 1);
			}
			switch (parts[0])
			{
				case "ping": return CmdPing();
				case "accounts": return CmdAccounts();
				case "account": return AccountJson(RequireAccount(Arg(a, "account")));
				case "instrument": return CmdInstrument(a);
				case "quote": return CmdQuote(a);
				case "bars": return CmdBars(a);
				case "flow": return CmdFlow(a);
				case "positions": return CmdPositions(a);
				case "tickets": return CmdTickets(a);
				case "enter": return CmdEnter(a);
				case "close": return CmdClose(a);
				case "modify": return CmdModify(a);
				case "cancel": return CmdCancel(a);
				case "flatten": return CmdFlatten(a);
				default: throw new Exception("unknown command '" + parts[0] + "'");
			}
		}

		// ------------------------------------------------------------------
		// commands: information
		// ------------------------------------------------------------------

		private string CmdPing()
		{
			return J.Obj("version", J.Str(BridgeVersion), "utc", J.Num(Epoch(DateTime.UtcNow)),
				"connections", J.Arr(ConnectionStates()));
		}

		private string CmdAccounts()
		{
			List<string> items = new List<string>();
			List<Account> all = new List<Account>();
			lock (Account.All)
				all.AddRange(Account.All);
			foreach (Account acct in all)
				items.Add(AccountJson(acct));
			return J.Obj("accounts", J.Arr(items));
		}

		private static string AccountJson(Account acct)
		{
			string provider = Convert.ToString(Prop(acct, "Provider"));
			object conn = Prop(acct, "Connection");
			string connName = conn == null ? "none" : Convert.ToString(Prop(Prop(conn, "Options"), "Name"));
			string status = conn == null ? "Unknown" : Convert.ToString(Prop(conn, "Status"));
			bool isSim = provider == "Simulator"
				|| acct.Name.StartsWith("Sim", StringComparison.OrdinalIgnoreCase)
				|| acct.Name.StartsWith("Playback", StringComparison.OrdinalIgnoreCase);
			return J.Obj(
				"name", J.Str(acct.Name),
				"provider", J.Str(provider),
				"connection", J.Str(connName),
				"status", J.Str(status),
				"connected", J.Bool(status == "Connected"),
				"is_sim", J.Bool(isSim),
				"cash", J.Num(SafeGet(acct, AccountItem.CashValue)),
				"realized", J.Num(SafeGet(acct, AccountItem.RealizedProfitLoss)),
				"unrealized", J.Num(SafeGet(acct, AccountItem.UnrealizedProfitLoss)));
		}

		private string CmdInstrument(Dictionary<string, string> a)
		{
			Instrument inst = RequireInstrument(Arg(a, "name"));
			MasterInstrument mi = inst.MasterInstrument;
			object expiry = Prop(inst, "Expiry");
			return J.Obj(
				"name", J.Str(inst.FullName),
				"root", J.Str(mi.Name),
				"point_value", J.Num(mi.PointValue),
				"tick_size", J.Num(mi.TickSize),
				"type", J.Str(mi.InstrumentType.ToString()),
				"expiry", J.Str(expiry is DateTime ? ((DateTime)expiry).ToString("yyyy-MM-dd", CultureInfo.InvariantCulture) : ""));
		}

		private string CmdQuote(Dictionary<string, string> a)
		{
			Instrument inst = RequireInstrument(Arg(a, "instrument"));
			Quote q = EnsureFeed(inst);
			double age = q.Utc == DateTime.MinValue ? double.NaN : (DateTime.UtcNow - q.Utc).TotalSeconds;
			return J.Obj("instrument", J.Str(inst.FullName), "bid", J.Num(q.Bid), "ask", J.Num(q.Ask),
				"last", J.Num(q.Last), "age_s", J.Num(age));
		}

		private string CmdBars(Dictionary<string, string> a)
		{
			Instrument inst = RequireInstrument(Arg(a, "instrument"));
			int minutes = ArgI(a, "minutes");
			int count = ArgI(a, "count");
			if (minutes <= 0 || count <= 0)
				throw new Exception("minutes and count must be positive");
			bool daily = minutes >= 1440;

			BarsPeriod period = new BarsPeriod();
			period.BarsPeriodType = daily ? BarsPeriodType.Day : BarsPeriodType.Minute;
			period.Value = daily ? 1 : minutes;

			DateTime nowUtc = DateTime.UtcNow;
			TimeZoneInfo et = TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time");
			DateTime tradeDate = TimeZoneInfo.ConvertTimeFromUtc(nowUtc, et).AddHours(7).Date;

			List<string> rows = new List<string>();
			string error = null;
			ManualResetEventSlim done = new ManualResetEventSlim(false);
			BarsRequest req = new BarsRequest(inst, count + 2);
			req.BarsPeriod = period;
			req.TradingHours = inst.MasterInstrument.TradingHours;
			req.Request((r, code, msg) =>
			{
				try
				{
					if (code != ErrorCode.NoError)
					{
						error = code + " " + msg;
						return;
					}
					Bars b = r.Bars;
					for (int i = 0; i < b.Count; i++)
					{
						DateTime t = b.GetTime(i);
						DateTime openUtc;
						if (daily)
						{
							// Day bars carry the trade date; today's is still forming.
							if (t.Date >= tradeDate)
								continue;
							openUtc = DateTime.SpecifyKind(t.Date, DateTimeKind.Utc);
						}
						else
						{
							// NinjaTrader stamps bars at their CLOSE, in its own time zone.
							DateTime closeUtc = ToUtc(t);
							if (closeUtc > nowUtc.AddSeconds(1))
								continue;
							openUtc = closeUtc.AddMinutes(-minutes);
						}
						rows.Add("[" + J.Num(Epoch(openUtc)) + "," + J.Num(b.GetOpen(i)) + "," + J.Num(b.GetHigh(i)) + ","
							+ J.Num(b.GetLow(i)) + "," + J.Num(b.GetClose(i)) + "," + J.Num(b.GetVolume(i)) + "]");
					}
				}
				catch (Exception ex)
				{
					error = ex.Message;
				}
				finally
				{
					done.Set();
				}
			});
			bool finished = done.Wait(30000);
			IDisposable disposable = req as IDisposable;
			if (disposable != null)
			{
				try { disposable.Dispose(); } catch { }
			}
			if (!finished)
				throw new Exception("bars request for " + inst.FullName + " timed out (is the data connection up?)");
			if (error != null)
				throw new Exception("bars request for " + inst.FullName + " failed: " + error);
			if (rows.Count > count)
				rows = rows.GetRange(rows.Count - count, count);
			return J.Obj("instrument", J.Str(inst.FullName), "bars", J.Arr(rows));
		}

		// Order flow for research: every trade in [from, to) (UTC epoch seconds, at most 3 days),
		// summed per minute inside NinjaTrader -- a day of NQ is up to a million trades, the
		// reply ~1,400 rows of [minute, buy volume, sell volume, unclassified volume, trades,
		// big-trade buy volume, big-trade sell volume] (big = at least `big` contracts, default 10).
		// A trade at or above the ask is BUYING (a buyer crossed the spread), at or below the bid
		// SELLING. Inside the spread, or when the history has no bid/ask, the tick rule decides:
		// above the last trade = buying, below = selling, same price = as the trade before.
		// "with_quote" says how many trades had a usable bid/ask.
		private string CmdFlow(Dictionary<string, string> a)
		{
			Instrument inst = RequireInstrument(Arg(a, "instrument"));
			DateTime fromUtc = UnixEpoch.AddSeconds(ArgD(a, "from"));
			DateTime toUtc = UnixEpoch.AddSeconds(ArgD(a, "to"));
			string bigArg;
			int big = a.TryGetValue("big", out bigArg) && !string.IsNullOrEmpty(bigArg) ? ArgI(a, "big") : 10;
			if (toUtc <= fromUtc || (toUtc - fromUtc).TotalDays > 3)
				throw new Exception("flow needs from < to, at most 3 days apart");
			TimeZoneInfo tz = NtZone();
			DateTime fromLocal = TimeZoneInfo.ConvertTimeFromUtc(fromUtc, tz);
			DateTime toLocal = TimeZoneInfo.ConvertTimeFromUtc(toUtc, tz);
			double maxSpread = 40 * inst.MasterInstrument.TickSize;

			BarsPeriod period = new BarsPeriod();
			period.BarsPeriodType = BarsPeriodType.Tick;
			period.Value = 1;

			SortedDictionary<long, long[]> mins = new SortedDictionary<long, long[]>();
			long ticks = 0;
			long withQuote = 0;
			string error = null;
			ManualResetEventSlim done = new ManualResetEventSlim(false);
			BarsRequest req = new BarsRequest(inst, fromLocal, toLocal);
			req.BarsPeriod = period;
			req.TradingHours = inst.MasterInstrument.TradingHours;
			req.Request((r, code, msg) =>
			{
				try
				{
					if (code != ErrorCode.NoError)
					{
						error = code + " " + msg;
						return;
					}
					Bars b = r.Bars;
					double lastPx = double.NaN;
					int lastSide = 0;
					for (int i = 0; i < b.Count; i++)
					{
						DateTime t = ToUtc(b.GetTime(i));
						if (t < fromUtc || t >= toUtc)
							continue;
						double px = b.GetClose(i);
						long vol = b.GetVolume(i);
						double bid = b.GetBid(i);
						double ask = b.GetAsk(i);
						int side = 0;
						// A real quote has a spread: bid = ask = trade price means "no bid/ask history".
						if (bid > 0 && ask > bid && ask - bid <= maxSpread)
						{
							withQuote++;
							if (px >= ask)
								side = 1;
							else if (px <= bid)
								side = -1;
						}
						if (side == 0)
						{
							if (!double.IsNaN(lastPx) && px > lastPx)
								side = 1;
							else if (!double.IsNaN(lastPx) && px < lastPx)
								side = -1;
							else
								side = lastSide;
						}
						lastPx = px;
						lastSide = side;
						ticks++;
						long minute = (long)Math.Floor(Epoch(t) / 60.0) * 60;
						long[] row;
						if (!mins.TryGetValue(minute, out row))
						{
							row = new long[6];
							mins[minute] = row;
						}
						if (side > 0)
							row[0] += vol;
						else if (side < 0)
							row[1] += vol;
						else
							row[2] += vol;
						row[3]++;
						if (vol >= big)
						{
							if (side > 0)
								row[4] += vol;
							else if (side < 0)
								row[5] += vol;
						}
					}
				}
				catch (Exception ex)
				{
					error = ex.Message;
				}
				finally
				{
					done.Set();
				}
			});
			bool finished = done.Wait(170000);
			IDisposable disposable = req as IDisposable;
			if (disposable != null)
			{
				try { disposable.Dispose(); } catch { }
			}
			if (!finished)
				throw new Exception("flow request for " + inst.FullName + " timed out (is the data connection up?)");
			if (error != null)
				throw new Exception("flow request for " + inst.FullName + " failed: " + error);
			List<string> rows = new List<string>(mins.Count);
			foreach (KeyValuePair<long, long[]> kv in mins)
			{
				long[] v = kv.Value;
				rows.Add("[" + J.Int(kv.Key) + "," + J.Int(v[0]) + "," + J.Int(v[1]) + "," + J.Int(v[2]) + ","
					+ J.Int(v[3]) + "," + J.Int(v[4]) + "," + J.Int(v[5]) + "]");
			}
			return J.Obj("instrument", J.Str(inst.FullName), "ticks", J.Int(ticks), "with_quote", J.Int(withQuote),
				"minutes", J.Arr(rows));
		}

		private string CmdPositions(Dictionary<string, string> a)
		{
			Account acct = RequireAccount(Arg(a, "account"));
			List<string> items = new List<string>();
			lock (acct.Positions)
			{
				foreach (Position p in acct.Positions)
				{
					if (p.MarketPosition == MarketPosition.Flat || p.Quantity == 0)
						continue;
					items.Add(J.Obj(
						"instrument", J.Str(p.Instrument.FullName),
						"side", J.Str(p.MarketPosition == MarketPosition.Long ? "long" : "short"),
						"qty", J.Int(p.Quantity),
						"avg_price", J.Num(p.AveragePrice)));
				}
			}
			return J.Obj("positions", J.Arr(items));
		}

		private string CmdTickets(Dictionary<string, string> a)
		{
			string acctName = null;
			if (a.ContainsKey("account"))
			{
				acctName = a["account"];
				RequireAccount(acctName);            // hooks the account and rebuilds after a restart
			}
			List<Ticket> list;
			lock (sync)
				list = tickets.Values.Where(t => acctName == null || t.AccountName == acctName).ToList();
			return J.Obj("tickets", J.Arr(list.Select(t => TicketJson(t))));
		}

		// ------------------------------------------------------------------
		// commands: trading
		// ------------------------------------------------------------------

		private string CmdEnter(Dictionary<string, string> a)
		{
			long id = ArgL(a, "ticket");
			lock (sync)
			{
				Ticket existing;
				if (tickets.TryGetValue(id, out existing))
					return TicketJson(existing);     // a retried request: never place it twice
			}
			Account acct = RequireAccount(Arg(a, "account"));
			Instrument inst = RequireInstrument(Arg(a, "instrument"));
			string side = Arg(a, "side");
			if (side != "long" && side != "short")
				throw new Exception("side must be long or short");
			int qty = ArgI(a, "qty");
			double sl = ArgD(a, "sl_dist");
			double tp = ArgD(a, "tp_dist");
			int magic = ArgI(a, "magic");
			if (qty < 1)
				throw new Exception("qty must be at least 1");
			if (sl <= 0 || tp <= 0)
				throw new Exception("sl_dist and tp_dist must be positive");

			lock (sync)
			{
				foreach (Ticket o in tickets.Values)
				{
					if (o.AccountName == acct.Name && o.InstrumentName == inst.FullName && !IsDone(o))
						throw new Exception(inst.FullName + " already has an open bot ticket (" + o.Id + ")");
				}
			}
			if (SignedPosition(acct, inst.FullName) != 0)
				throw new Exception("the account already holds a position in " + inst.FullName);

			EnsureFeed(inst);
			Ticket t = new Ticket();
			t.Id = id;
			t.Magic = magic;
			t.AccountName = acct.Name;
			t.InstrumentName = inst.FullName;
			t.IsLong = side == "long";
			t.Qty = qty;
			t.SlDist = sl;
			t.TpDist = tp;
			t.OpenedUtc = DateTime.UtcNow;
			Order entry = acct.CreateOrder(inst, t.IsLong ? OrderAction.Buy : OrderAction.SellShort, OrderType.Market,
				OrderEntry.Automated, TimeInForce.Day, qty, 0, 0, string.Empty, NameFor(t, "E"), Core.Globals.MaxDate, null);
			t.Entry = entry;
			lock (sync)
				tickets[id] = t;
			acct.Submit(new[] { entry });
			return TicketJson(t);
		}

		private string CmdClose(Dictionary<string, string> a)
		{
			Ticket t = RequireTicket(a);
			Account acct = RequireAccount(t.AccountName);
			Instrument inst = RequireInstrument(t.InstrumentName);
			string reason = a.ContainsKey("reason") ? a["reason"] : "manual";
			int want = a.ContainsKey("qty") ? ArgI(a, "qty") : int.MaxValue;

			// Nothing may be added to a position that is being closed.
			if (t.Entry != null && !IsTerminal(t.Entry.OrderState))
			{
				acct.Cancel(new[] { t.Entry });
				WaitTerminal(t, new List<Order> { t.Entry }, 5000);
			}
			if (AnyExitWorking(t))
				return J.Obj("submitted", "0", "pending", "true");
			// Never close more than the account actually holds this ticket's way
			// (see Audit step 2) -- an exit for contracts that are already gone
			// would open a position in the opposite direction.
			int open = Held(SignedPosition(acct, t.InstrumentName), t, OpenQty(t));
			if (open <= 0)
				return J.Obj("submitted", "0", "pending", "false");

			if (want < open)
			{
				// Shrink the protection to what will remain, then exit the difference.
				ResizeProtection(t, acct, open - want);
				SubmitExit(t, acct, inst, want);
				return J.Obj("submitted", J.Int(want), "pending", "false");
			}

			// Full close: cancel the protection FIRST and wait until it is dead,
			// so a stop filling in the gap cannot combine with our exit into a
			// reversed position.
			lock (sync)
				t.Closing = true;
			List<Order> protect = WorkingProtection(t);
			if (protect.Count > 0)
			{
				acct.Cancel(protect.ToArray());
				if (!WaitTerminal(t, protect, 5000))
				{
					lock (sync)
						t.Closing = false;             // let the audit keep guarding it
					throw new Exception("protective orders did not cancel within 5s -- exit not sent, retry");
				}
			}
			open = Held(SignedPosition(acct, t.InstrumentName), t, OpenQty(t));
			if (open <= 0)
			{
				// Nothing left at the broker: let the audit book what closed it.
				lock (sync)
					t.Closing = false;
				return J.Obj("submitted", "0", "pending", "false");
			}
			lock (sync)
			{
				if (string.IsNullOrEmpty(t.CloseReason))
					t.CloseReason = reason;
			}
			SubmitExit(t, acct, inst, open);
			return J.Obj("submitted", J.Int(open), "pending", "false");
		}

		private string CmdModify(Dictionary<string, string> a)
		{
			Ticket t = RequireTicket(a);
			Account acct = RequireAccount(t.AccountName);
			Instrument inst = RequireInstrument(t.InstrumentName);
			MasterInstrument mi = inst.MasterInstrument;
			double sl = ArgD(a, "sl");
			double tp = ArgD(a, "tp");
			Quote q = EnsureFeed(inst);
			List<Order> change = new List<Order>();

			Order stop = t.Stop;
			if (sl > 0 && stop != null && !IsTerminal(stop.OrderState))
			{
				double px = mi.RoundToTickSize(sl);
				// A stop on the wrong side of the market would fill at once.
				bool wrongSide = q.HasBidAsk && (t.IsLong ? px >= q.Bid : px <= q.Ask);
				if (!wrongSide && Math.Abs(px - stop.StopPrice) > mi.TickSize / 2)
				{
					stop.StopPriceChanged = px;
					change.Add(stop);
				}
			}
			Order target = t.Target;
			if (tp > 0 && target != null && !IsTerminal(target.OrderState))
			{
				double px = mi.RoundToTickSize(tp);
				bool wrongSide = q.HasBidAsk && (t.IsLong ? px <= q.Bid : px >= q.Ask);
				if (!wrongSide && Math.Abs(px - target.LimitPrice) > mi.TickSize / 2)
				{
					target.LimitPriceChanged = px;
					change.Add(target);
				}
			}
			if (change.Count > 0)
				acct.Change(change.ToArray());
			return J.Obj("changed", J.Bool(change.Count > 0));
		}

		// NinjaTrader's own flatten for one contract: cancels its working orders
		// and closes the whole position, whoever opened it. For cleanup and
		// emergencies; the audit books any bot ticket it closes as "external".
		private string CmdFlatten(Dictionary<string, string> a)
		{
			Account acct = RequireAccount(Arg(a, "account"));
			Instrument inst = RequireInstrument(Arg(a, "instrument"));
			int before = SignedPosition(acct, inst.FullName);
			acct.Flatten(new[] { inst });
			return J.Obj("instrument", J.Str(inst.FullName), "position_before", J.Int(before));
		}

		private string CmdCancel(Dictionary<string, string> a)
		{
			Ticket t = RequireTicket(a);
			Account acct = RequireAccount(t.AccountName);
			if (t.Entry != null && !IsTerminal(t.Entry.OrderState))
				acct.Cancel(new[] { t.Entry });
			return TicketJson(t);
		}

		// ------------------------------------------------------------------
		// order events: bracket placement
		// ------------------------------------------------------------------

		private void OnOrderUpdate(object sender, OrderEventArgs e)
		{
			try
			{
				Order o = e.Order;
				long id;
				int magic;
				string role;
				if (o == null || !TryParseName(o.Name, out id, out magic, out role))
					return;
				Ticket t;
				lock (sync)
				{
					if (!tickets.TryGetValue(id, out t))
						return;
					Attach(t, o, role);
					if (role != "E" && o.Filled > 0 && (o.OrderState == OrderState.Filled || o.OrderState == OrderState.PartFilled))
						fillTimes[o.Name] = DateTime.UtcNow;
				}
				if (role == "E")
				{
					ThreadPool.QueueUserWorkItem(delegate { Safe(delegate { OnEntryUpdate(t, o); }); });
				}
				else if (role == "S" && o.OrderState == OrderState.Rejected)
				{
					string why = Convert.ToString(Prop(e, "Comment"));
					ThreadPool.QueueUserWorkItem(delegate { Safe(delegate { OnStopRejected(t, why); }); });
				}
			}
			catch (Exception ex)
			{
				Print("VertexBridge OnOrderUpdate: " + ex.Message);
			}
		}

		private void OnEntryUpdate(Ticket t, Order entry)
		{
			int filled = entry.Filled;
			if (filled <= 0)
			{
				if (IsTerminal(entry.OrderState))
				{
					lock (sync)
					{
						t.BracketState = "none";
						if (entry.OrderState == OrderState.Rejected)
							t.Error = "entry rejected by the broker";
					}
				}
				return;
			}
			bool send = false;
			bool resize = false;
			lock (sync)
			{
				if (!t.BracketSent)
				{
					t.BracketSent = true;
					t.ProtectedQty = filled;
					send = true;
				}
				else if (filled > t.ProtectedQty)
				{
					t.ProtectedQty = filled;
					resize = true;
				}
			}
			Account acct = FindHooked(t.AccountName);
			if (acct == null)
				return;
			if (send)
				SubmitBracket(t, acct, entry);
			else if (resize)
				ResizeProtection(t, acct, filled);
		}

		private void SubmitBracket(Ticket t, Account acct, Order entry)
		{
			Instrument inst = RequireInstrument(t.InstrumentName);
			MasterInstrument mi = inst.MasterInstrument;
			int qty = Math.Max(1, entry.Filled);
			double fill = entry.AverageFillPrice;
			double stop = mi.RoundToTickSize(t.IsLong ? fill - t.SlDist : fill + t.SlDist);
			double target = mi.RoundToTickSize(t.IsLong ? fill + t.TpDist : fill - t.TpDist);

			// If the market already traded through the stop while the entry was
			// filling, a stop order would be rejected -- get out now instead.
			Quote q;
			lock (sync)
				quotes.TryGetValue(t.InstrumentName, out q);
			if (q != null && q.HasBidAsk && (t.IsLong ? q.Bid <= stop : q.Ask >= stop))
			{
				lock (sync)
				{
					t.BracketState = "failed";
					t.Closing = true;
					t.Error = "price already beyond the stop when the entry filled -- closed at market";
					t.CloseReason = "gap";
				}
				SubmitExit(t, acct, inst, qty);
				return;
			}

			OrderAction exit = t.IsLong ? OrderAction.Sell : OrderAction.BuyToCover;
			string oco = "VXOCO" + t.Id.ToString(CultureInfo.InvariantCulture) + "_" + (++t.OcoSeq).ToString(CultureInfo.InvariantCulture);
			Order s = acct.CreateOrder(inst, exit, OrderType.StopMarket, OrderEntry.Automated, TimeInForce.Day, qty, 0, stop,
				oco, NameFor(t, "S"), Core.Globals.MaxDate, null);
			Order tp = acct.CreateOrder(inst, exit, OrderType.Limit, OrderEntry.Automated, TimeInForce.Day, qty, target, 0,
				oco, NameFor(t, "T"), Core.Globals.MaxDate, null);
			lock (sync)
			{
				t.Stop = s;
				t.Target = tp;
				t.Exits[s.Name] = s;
				t.Exits[tp.Name] = tp;
				t.BracketState = "working";
			}
			acct.Submit(new[] { s, tp });
		}

		private void OnStopRejected(Ticket t, string why)
		{
			Account acct = FindHooked(t.AccountName);
			if (acct == null)
				return;
			lock (sync)
			{
				if (t.Closing || t.Rebuilt)      // rebuilt tickets never get automatic orders
					return;
				t.Closing = true;
				t.BracketState = "failed";
				t.Error = "stop order rejected (" + why + ") -- position closed at market";
				t.CloseReason = "protection_lost";
			}
			List<Order> protect = WorkingProtection(t);
			if (protect.Count > 0)
				acct.Cancel(protect.ToArray());
			int open = OpenQty(t);
			if (open > 0)
				SubmitExit(t, acct, RequireInstrument(t.InstrumentName), open);
		}

		private void SubmitExit(Ticket t, Account acct, Instrument inst, int qty)
		{
			string name;
			lock (sync)
				name = NameFor(t, "X" + (++t.ExitSeq).ToString(CultureInfo.InvariantCulture));   // X1, X2, ...: each exit uniquely named
			Order x = acct.CreateOrder(inst, t.IsLong ? OrderAction.Sell : OrderAction.BuyToCover, OrderType.Market,
				OrderEntry.Automated, TimeInForce.Day, qty, 0, 0, string.Empty, name, Core.Globals.MaxDate, null);
			lock (sync)
				t.Exits[name] = x;
			acct.Submit(new[] { x });
		}

		private void ResizeProtection(Ticket t, Account acct, int qty)
		{
			List<Order> protect = WorkingProtection(t);
			if (protect.Count == 0)
				return;
			if (qty <= 0)
			{
				acct.Cancel(protect.ToArray());
				return;
			}
			// An order's Quantity includes what it has already filled, so the
			// size that makes `qty` contracts REMAIN is Filled + qty.
			List<Order> change = new List<Order>();
			foreach (Order o in protect)
			{
				// 1.2 (2026-10-01): changing an order that is not accepted yet makes
				// NinjaTrader REJECT it ("can't be submitted: order status is
				// ChangePending"); the OCO partner is then cancelled and the position
				// is left unprotected. A 2-lot market entry that filled 1 + 1 hit this
				// within 0.1 s. Such orders are skipped here; the audit resizes them
				// once they are working.
				if (!Changeable(o))
					continue;
				int desired = o.Filled + qty;
				if (o.Quantity != desired)
				{
					o.QuantityChanged = desired;
					change.Add(o);
				}
			}
			if (change.Count > 0)
				acct.Change(change.ToArray());
		}

		// ------------------------------------------------------------------
		// audit: runs every 2 s, keeps every open ticket protected and honest
		// ------------------------------------------------------------------

		private void AuditTick(object state)
		{
			if (Interlocked.Exchange(ref auditBusy, 1) == 1)
				return;
			try
			{
				Audit();
			}
			catch (Exception ex)
			{
				Print("VertexBridge audit: " + ex.Message);
			}
			finally
			{
				Interlocked.Exchange(ref auditBusy, 0);
			}
		}

		private void Audit()
		{
			List<Ticket> list;
			lock (sync)
				list = tickets.Values.Where(x => !x.Finalized).ToList();
			foreach (Ticket t in list)
			{
				Account acct = FindHooked(t.AccountName);
				if (acct == null || t.Entry == null)
					continue;     // rebuilt without its entry: leave the broker's orders alone
				int entryFilled = t.Entry.Filled;
				bool entryDone = IsTerminal(t.Entry.OrderState);
				int open = OpenQty(t);
				int acctPos = SignedPosition(acct, t.InstrumentName);

				// 1. Finished: cancel anything left, or it could open a NEW
				// position later. Only when the account is flat (or the other
				// way): if it still holds contracts in this ticket's direction
				// our accounting may have missed a fill, and cancelling would
				// strip a live position of its stop.
				if (entryDone && (entryFilled == 0 || open <= 0))
				{
					List<Order> leftovers = WorkingProtection(t);
					bool ownDirection = t.IsLong ? acctPos > 0 : acctPos < 0;
					if (leftovers.Count > 0 && !ownDirection)
						acct.Cancel(leftovers.ToArray());
					lock (sync)
					{
						if (!t.ClosedUtc.HasValue)
							t.ClosedUtc = LastExitTime(t);
						if (leftovers.Count > 0 && ownDirection)
							t.Error = "closed by our accounting but the account still holds " + acctPos + " -- orders left in place, check NinjaTrader";
						t.Finalized = leftovers.Count == 0;
					}
					continue;
				}
				if (open <= 0 || t.Closing || AnyExitWorking(t))
				{
					t.Suspect = 0;
					t.Mismatch = 0;
					t.SizeDrift = 0;
					continue;
				}

				// 2. Reconcile with what the account REALLY holds before anything
				// else. Our bookkeeping can be wrong -- after a restart or
				// recompile, a missed fill, an order we no longer recognise -- and
				// on 2026-09-29 exactly that made this audit "protect" a position
				// that was already closed by SELLING it, opening a short. So: when
				// the account holds less than we think, only the books change,
				// and no order below is ever sent while the two disagree.
				int held = Held(acctPos, t, open);
				int gone = open - held;
				if (gone > 0)
				{
					// 3 audits (4-6 s) so an account update that merely lags a
					// fill is never mistaken for an outside close.
					if (++t.Mismatch >= 3)
					{
						double px = ExternalExitPrice(acct, t);
						lock (sync)
						{
							t.ExternalExitQty += gone;
							t.ExternalExitNotional += gone * px;
							if (held == 0 && string.IsNullOrEmpty(t.CloseReason))
								t.CloseReason = "external";
						}
						ResizeProtection(t, acct, held);
						Print("VertexBridge: ticket " + t.Id + " -- " + gone + " contract(s) of " + t.InstrumentName
							+ " closed outside the bot at ~" + px.ToString(CultureInfo.InvariantCulture));
						t.Mismatch = 0;
					}
					t.Suspect = 0;
					t.SizeDrift = 0;
					continue;
				}
				t.Mismatch = 0;

				// 3. A live stop of the right size. Only for tickets this session
				// opened itself: a ticket rebuilt after a restart gets bookkeeping
				// only, never an automatic order.
				if (t.Rebuilt || t.BracketState != "working")
					continue;
				Order stop = t.Stop;
				bool stopAlive = stop != null && !IsTerminal(stop.OrderState);
				if (!stopAlive)
				{
					if (++t.Suspect >= 2)
					{
						lock (sync)
						{
							t.Closing = true;
							t.BracketState = "failed";
							t.Error = "stop order no longer working -- position closed at market";
							if (string.IsNullOrEmpty(t.CloseReason))
								t.CloseReason = "protection_lost";
						}
						List<Order> rest = WorkingProtection(t);
						if (rest.Count > 0)
							acct.Cancel(rest.ToArray());
						SubmitExit(t, acct, RequireInstrument(t.InstrumentName), held);   // held == open here
					}
					continue;
				}
				t.Suspect = 0;
				int stopRemaining = stop.Quantity - stop.Filled;
				Order tgt = t.Target;
				bool tgtAlive = tgt != null && !IsTerminal(tgt.OrderState);
				bool drift = stopRemaining != open || (tgtAlive && tgt.Quantity - tgt.Filled != open);
				bool settling = !Changeable(stop) || (tgtAlive && !Changeable(tgt));
				if (drift && !settling)
				{
					if (++t.SizeDrift >= 2)
					{
						ResizeProtection(t, acct, open);
						t.SizeDrift = 0;
					}
				}
				else
					t.SizeDrift = 0;
			}
		}

		// Contracts of this ticket the account actually holds: the part of the
		// account's position that points the ticket's way, never more than `open`.
		private static int Held(int acctPos, Ticket t, int open)
		{
			int mine = t.IsLong ? open : -open;
			return Math.Sign(acctPos) == Math.Sign(mine) ? Math.Min(Math.Abs(acctPos), open) : 0;
		}

		// ------------------------------------------------------------------
		// ticket bookkeeping
		// ------------------------------------------------------------------

		// States in which NinjaTrader accepts a change request.
		private static bool Changeable(Order o)
		{
			OrderState s = o.OrderState;
			return s == OrderState.Accepted || s == OrderState.Working || s == OrderState.TriggerPending
				|| s == OrderState.PartFilled;
		}

		private static bool IsTerminal(OrderState s)
		{
			return s == OrderState.Filled || s == OrderState.Cancelled || s == OrderState.Rejected || s == OrderState.Unknown;
		}

		// "S", "T" or "X" (every X1, X2, ... exit counts as "X").
		private static bool IsRole(Order o, string role)
		{
			if (o == null || o.Name == null)
				return false;
			int bar = o.Name.LastIndexOf('|');
			string r = bar < 0 ? "" : o.Name.Substring(bar + 1);
			return role == "X" ? r.StartsWith("X", StringComparison.Ordinal) : r == role;
		}

		private List<Order> WorkingProtection(Ticket t)
		{
			lock (sync)
				return t.Exits.Values.Where(o => (IsRole(o, "S") || IsRole(o, "T")) && !IsTerminal(o.OrderState)).ToList();
		}

		private bool AnyExitWorking(Ticket t)
		{
			lock (sync)
				return t.Exits.Values.Any(o => IsRole(o, "X") && !IsTerminal(o.OrderState));
		}

		private void ExitStats(Ticket t, out int qty, out double avg)
		{
			double notional;
			lock (sync)
			{
				qty = t.ExternalExitQty;
				notional = t.ExternalExitNotional;
				foreach (Order o in t.Exits.Values)
				{
					if (o.Filled <= 0)
						continue;
					qty += o.Filled;
					notional += o.Filled * o.AverageFillPrice;
				}
			}
			avg = qty > 0 ? notional / qty : double.NaN;
		}

		private int OpenQty(Ticket t)
		{
			if (t.Entry == null)
				return 0;
			int exited;
			double avg;
			ExitStats(t, out exited, out avg);
			return Math.Max(0, t.Entry.Filled - exited);
		}

		private bool IsDone(Ticket t)
		{
			// Rebuilt without its entry: not manageable, so it must not block the
			// instrument. A real position still blocks entries via SignedPosition.
			if (t.Entry == null)
				return true;
			return IsTerminal(t.Entry.OrderState) && (t.Entry.Filled == 0 || OpenQty(t) <= 0);
		}

		private DateTime LastExitTime(Ticket t)
		{
			DateTime last = DateTime.MinValue;
			lock (sync)
			{
				foreach (string name in t.Exits.Keys)
				{
					DateTime when;
					if (fillTimes.TryGetValue(name, out when) && when > last)
						last = when;
				}
			}
			return last == DateTime.MinValue ? DateTime.UtcNow : last;
		}

		private string CloseReason(Ticket t)
		{
			lock (sync)
			{
				if (!string.IsNullOrEmpty(t.CloseReason))
					return t.CloseReason;
				int stopQty = t.Exits.Values.Where(o => IsRole(o, "S")).Sum(o => o.Filled);
				int targetQty = t.Exits.Values.Where(o => IsRole(o, "T")).Sum(o => o.Filled);
				if (stopQty == 0 && targetQty == 0)
					return t.ExternalExitQty > 0 ? "external" : "manual";
				return stopQty >= targetQty ? "sl" : "tp";
			}
		}

		private string TicketJson(Ticket t)
		{
			Order entry = t.Entry;
			int entryFilled = entry == null ? 0 : entry.Filled;
			string entryState = entry == null ? "Unknown" : entry.OrderState.ToString();
			int exited;
			double exitAvg;
			ExitStats(t, out exited, out exitAvg);
			int open = entry == null ? 0 : Math.Max(0, entryFilled - exited);
			bool closed = entry != null && IsTerminal(entry.OrderState) && (entryFilled == 0 || open == 0);
			DateTime? closedUtc = t.ClosedUtc;
			if (closed && !closedUtc.HasValue)
				closedUtc = LastExitTime(t);
			Quote q;
			lock (sync)
				quotes.TryGetValue(t.InstrumentName, out q);
			double pv = double.NaN;
			try { pv = Instrument.GetInstrument(t.InstrumentName).MasterInstrument.PointValue; } catch { }
			Order stop = t.Stop;
			Order target = t.Target;
			return J.Obj(
				"ticket", J.Int(t.Id),
				"magic", J.Int(t.Magic),
				"account", J.Str(t.AccountName),
				"instrument", J.Str(t.InstrumentName),
				"side", J.Str(t.IsLong ? "long" : "short"),
				"qty", J.Int(t.Qty),
				"entry_state", J.Str(entryState),
				"entry_filled", J.Int(entryFilled),
				"entry_avg", J.Num(entryFilled > 0 ? entry.AverageFillPrice : double.NaN),
				"open_qty", J.Int(open),
				"closed_qty", J.Int(exited),
				"exit_avg", J.Num(exited > 0 ? exitAvg : double.NaN),
				"closed", J.Bool(closed),
				"close_reason", J.Str(closed ? CloseReason(t) : ""),
				"closed_utc", closed ? J.Num(Epoch(closedUtc.Value)) : "null",
				"opened_utc", J.Num(Epoch(t.OpenedUtc)),
				"stop", J.Num(stop == null ? double.NaN : stop.StopPrice),
				"stop_state", J.Str(stop == null ? "" : stop.OrderState.ToString()),
				"target", J.Num(target == null ? double.NaN : target.LimitPrice),
				"target_state", J.Str(target == null ? "" : target.OrderState.ToString()),
				"bracket", J.Str(t.BracketState),
				"error", J.Str(t.Error),
				"point_value", J.Num(pv),
				"last", J.Num(q == null ? double.NaN : q.Last));
		}

		// Called under `sync`.
		private static void Attach(Ticket t, Order o, string role)
		{
			if (role == "E")
			{
				t.Entry = o;
				if (t.Qty == 0)
					t.Qty = o.Quantity;
				return;
			}
			t.Exits[o.Name] = o;             // newest instance replaces the old one
			if (role == "S")
				t.Stop = o;
			else if (role == "T")
				t.Target = o;
			else
			{
				int seq;
				if (int.TryParse(role.Substring(1), NumberStyles.Integer, CultureInfo.InvariantCulture, out seq) && seq > t.ExitSeq)
					t.ExitSeq = seq;         // after a rebuild, never reuse an exit name
			}
		}

		private void Rebuild(Account acct)
		{
			List<Order> orders = new List<Order>();
			lock (acct.Orders)
				orders.AddRange(acct.Orders);
			lock (sync)
			{
				foreach (Order o in orders)
				{
					long id;
					int magic;
					string role;
					if (!TryParseName(o.Name, out id, out magic, out role))
						continue;
					Ticket t;
					if (!tickets.TryGetValue(id, out t))
					{
						t = new Ticket();
						t.Id = id;
						t.Magic = magic;
						t.AccountName = acct.Name;
						t.InstrumentName = o.Instrument.FullName;
						t.OpenedUtc = ToUtc(o.Time);
						t.BracketSent = true;
						t.Rebuilt = true;
						tickets[id] = t;
					}
					Attach(t, o, role);
				}
				foreach (Ticket t in tickets.Values)
				{
					if (!t.Rebuilt || t.AccountName != acct.Name)
						continue;
					if (t.Entry != null)
					{
						t.IsLong = t.Entry.OrderAction == OrderAction.Buy;
						t.ProtectedQty = t.Entry.Filled;
					}
					else
					{
						Order x = t.Exits.Values.FirstOrDefault();
						if (x != null)
							t.IsLong = x.OrderAction == OrderAction.Sell;
						t.Error = "rebuilt without its entry order (NinjaTrader restarted?) -- manage this position manually";
					}
					t.BracketState = t.Stop != null ? "working" : "none";
				}
			}
		}

		private static string NameFor(Ticket t, string role)
		{
			return Prefix + t.Magic.ToString(CultureInfo.InvariantCulture) + "|" + t.Id.ToString(CultureInfo.InvariantCulture) + "|" + role;
		}

		private static bool TryParseName(string name, out long id, out int magic, out string role)
		{
			id = 0;
			magic = 0;
			role = null;
			if (string.IsNullOrEmpty(name) || !name.StartsWith(Prefix, StringComparison.Ordinal))
				return false;
			string[] p = name.Split('|');
			if (p.Length != 4)
				return false;
			if (!int.TryParse(p[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out magic))
				return false;
			if (!long.TryParse(p[2], NumberStyles.Integer, CultureInfo.InvariantCulture, out id))
				return false;
			role = p[3];
			// A bare "X" is the exit name used before exits were numbered.
			if (role == "E" || role == "S" || role == "T" || role == "X")
				return true;
			int seq;
			return role.Length > 1 && role[0] == 'X'
				&& int.TryParse(role.Substring(1), NumberStyles.Integer, CultureInfo.InvariantCulture, out seq);
		}

		// ------------------------------------------------------------------
		// accounts, instruments, positions, quotes
		// ------------------------------------------------------------------

		private Account RequireAccount(string name)
		{
			Account acct;
			lock (Account.All)
				acct = Account.All.FirstOrDefault(x => x.Name == name);
			if (acct == null)
				throw new Exception("account '" + name + "' not found in NinjaTrader (is its connection up?)");
			bool fresh;
			lock (sync)
			{
				fresh = !hooked.Contains(acct);
				if (fresh)
					hooked.Add(acct);
			}
			if (fresh)
			{
				acct.OrderUpdate += OnOrderUpdate;
				Rebuild(acct);
			}
			return acct;
		}

		private Account FindHooked(string name)
		{
			lock (sync)
				return hooked.FirstOrDefault(x => x.Name == name);
		}

		private static Instrument RequireInstrument(string name)
		{
			Instrument inst = Instrument.GetInstrument(name);
			if (inst == null)
				throw new Exception("instrument '" + name + "' not found in NinjaTrader's instrument database");
			return inst;
		}

		private Ticket RequireTicket(Dictionary<string, string> a)
		{
			long id = ArgL(a, "ticket");
			if (a.ContainsKey("account"))
				RequireAccount(a["account"]);
			lock (sync)
			{
				Ticket t;
				if (tickets.TryGetValue(id, out t))
					return t;
			}
			throw new Exception("unknown ticket " + id);
		}

		private static int SignedPosition(Account acct, string instrument)
		{
			lock (acct.Positions)
			{
				foreach (Position p in acct.Positions)
				{
					if (p.Instrument.FullName != instrument)
						continue;
					if (p.MarketPosition == MarketPosition.Long)
						return p.Quantity;
					if (p.MarketPosition == MarketPosition.Short)
						return -p.Quantity;
				}
			}
			return 0;
		}

		private double ExternalExitPrice(Account acct, Ticket t)
		{
			// Volume-weighted price of fills on this contract that the bot did
			// not place, on the side that closes this ticket.
			double qty = 0;
			double notional = 0;
			try
			{
				lock (acct.Executions)
				{
					foreach (Execution ex in acct.Executions)
					{
						if (ex.Instrument == null || ex.Instrument.FullName != t.InstrumentName)
							continue;
						if (ex.Order != null && ex.Order.Name != null && ex.Order.Name.StartsWith(Prefix, StringComparison.Ordinal))
							continue;
						bool isSell = ex.MarketPosition == MarketPosition.Short;
						if (t.IsLong != isSell || ToUtc(ex.Time) < t.OpenedUtc)
							continue;
						qty += ex.Quantity;
						notional += ex.Quantity * ex.Price;
					}
				}
			}
			catch (Exception e)
			{
				Print("VertexBridge: executions unreadable (" + e.Message + ")");
			}
			if (qty > 0)
				return notional / qty;
			Quote q;
			lock (sync)
				quotes.TryGetValue(t.InstrumentName, out q);
			if (q != null && !double.IsNaN(q.Last))
				return q.Last;
			return t.Entry != null ? t.Entry.AverageFillPrice : 0;
		}

		// 1.4 (2026-10-15): a price feed that has gone quiet is subscribed again. After NinjaTrader's restart
		// on 14 Oct the NQ feed delivered nothing for 10 hours (NinjaTrader logged "delayed data") while
		// ES and the minute bars kept coming -- the bots rightly refused to trade on the frozen price.
		private const double StaleQuoteS = 30.0;      // no tick for this long ...
		private const double ResubscribeS = 60.0;     // ... and the feed is at least this old: subscribe again

		private Quote EnsureFeed(Instrument inst)
		{
			string key = inst.FullName;
			MarketData md = null;
			MarketData old = null;
			Quote q;
			lock (sync)
			{
				DateTime started;
				if (quotes.TryGetValue(key, out q))
				{
					bool quiet = q.Utc == DateTime.MinValue || (DateTime.UtcNow - q.Utc).TotalSeconds > StaleQuoteS;
					if (!quiet || !feedStarted.TryGetValue(key, out started)
						|| (DateTime.UtcNow - started).TotalSeconds < ResubscribeS)
						return q;
					feeds.TryGetValue(key, out old);         // quiet for too long: replace the subscription
					if (old != null)
						feedKeys.Remove(old);
				}
				else
				{
					q = new Quote();
					quotes[key] = q;
				}
				md = new MarketData(inst);
				feeds[key] = md;
				feedKeys[md] = key;
				feedStarted[key] = DateTime.UtcNow;
			}
			if (old != null)
			{
				try { old.Update -= OnMarketData; } catch { }
			}
			md.Update += OnMarketData;
			return q;
		}

		private void OnMarketData(object sender, MarketDataEventArgs e)
		{
			try
			{
				Quote q = null;
				lock (sync)
				{
					string key = null;
					Instrument inst = Prop(e, "Instrument") as Instrument;
					if (inst != null)
						key = inst.FullName;
					else
					{
						MarketData md = sender as MarketData;
						if (md != null)
							feedKeys.TryGetValue(md, out key);
					}
					if (key == null || !quotes.TryGetValue(key, out q))
						return;
				}
				if (e.MarketDataType == MarketDataType.Bid)
					q.Bid = e.Price;
				else if (e.MarketDataType == MarketDataType.Ask)
					q.Ask = e.Price;
				else if (e.MarketDataType == MarketDataType.Last)
					q.Last = e.Price;
				else
					return;
				q.Utc = DateTime.UtcNow;
			}
			catch
			{
			}
		}

		private static List<string> ConnectionStates()
		{
			List<string> list = new List<string>();
			try
			{
				Type connType = typeof(Account).Assembly.GetType("NinjaTrader.Cbi.Connection");
				if (connType == null)
					return list;
				PropertyInfo p = connType.GetProperty("Connections", BindingFlags.Public | BindingFlags.Static);
				System.Collections.IEnumerable conns = p == null ? null : p.GetValue(null, null) as System.Collections.IEnumerable;
				if (conns == null)
					return list;
				foreach (object c in conns)
				{
					list.Add(J.Obj("name", J.Str(Convert.ToString(Prop(Prop(c, "Options"), "Name"))),
						"status", J.Str(Convert.ToString(Prop(c, "Status")))));
				}
			}
			catch
			{
			}
			return list;
		}

		// ------------------------------------------------------------------
		// helpers
		// ------------------------------------------------------------------

		private void Safe(Action action)
		{
			try
			{
				action();
			}
			catch (Exception ex)
			{
				Print("VertexBridge: " + ex.Message);
			}
		}

		// Waits on the NEWEST instance of each order (looked up by name), since
		// NinjaTrader may report state changes on a fresh Order object.
		private bool WaitTerminal(Ticket t, List<Order> orders, int timeoutMs)
		{
			DateTime deadline = DateTime.UtcNow.AddMilliseconds(timeoutMs);
			while (true)
			{
				bool allDone = true;
				lock (sync)
				{
					foreach (Order o in orders)
					{
						Order current = o;
						Order newer;
						if (IsRole(o, "S") || IsRole(o, "T") || IsRole(o, "X"))
						{
							if (t.Exits.TryGetValue(o.Name, out newer))
								current = newer;
						}
						else if (t.Entry != null)
							current = t.Entry;
						if (!IsTerminal(current.OrderState))
							allDone = false;
					}
				}
				if (allDone)
					return true;
				if (DateTime.UtcNow >= deadline)
					return false;
				Thread.Sleep(50);
			}
		}

		private static double SafeGet(Account acct, AccountItem item)
		{
			try
			{
				return acct.Get(item, Currency.UsDollar);
			}
			catch
			{
				return double.NaN;
			}
		}

		private static object Prop(object o, string name)
		{
			if (o == null)
				return null;
			try
			{
				PropertyInfo p = o.GetType().GetProperty(name, BindingFlags.Public | BindingFlags.Instance);
				return p == null ? null : p.GetValue(o, null);
			}
			catch
			{
				return null;
			}
		}

		private static TimeZoneInfo NtZone()
		{
			if (ntZone != null)
				return ntZone;
			TimeZoneInfo tz = null;
			try
			{
				// NinjaTrader stamps bars and executions in the time zone set under
				// Tools > Options > General. Read that setting; fall back to the PC's.
				Type g = typeof(NinjaTrader.Core.Globals);
				object opts = null;
				PropertyInfo p = g.GetProperty("GeneralOptions", BindingFlags.Public | BindingFlags.Static);
				if (p != null)
					opts = p.GetValue(null, null);
				else
				{
					FieldInfo f = g.GetField("GeneralOptions", BindingFlags.Public | BindingFlags.Static);
					if (f != null)
						opts = f.GetValue(null);
				}
				tz = Prop(opts, "TimeZoneInfo") as TimeZoneInfo;
			}
			catch
			{
			}
			ntZone = tz ?? TimeZoneInfo.Local;
			return ntZone;
		}

		private static DateTime ToUtc(DateTime t)
		{
			if (t.Kind == DateTimeKind.Utc)
				return t;
			TimeZoneInfo tz = NtZone();
			try
			{
				return TimeZoneInfo.ConvertTimeToUtc(DateTime.SpecifyKind(t, DateTimeKind.Unspecified), tz);
			}
			catch (ArgumentException)
			{
				// A wall-clock time skipped by a daylight-saving change.
				return DateTime.SpecifyKind(t - tz.GetUtcOffset(t), DateTimeKind.Utc);
			}
		}

		private static double Epoch(DateTime utc)
		{
			return (utc - UnixEpoch).TotalSeconds;
		}

		private static string Arg(Dictionary<string, string> a, string key)
		{
			string v;
			if (!a.TryGetValue(key, out v) || string.IsNullOrEmpty(v))
				throw new Exception("missing argument '" + key + "'");
			return v;
		}

		private static double ArgD(Dictionary<string, string> a, string key)
		{
			return double.Parse(Arg(a, key), NumberStyles.Float, CultureInfo.InvariantCulture);
		}

		private static int ArgI(Dictionary<string, string> a, string key)
		{
			return int.Parse(Arg(a, key), NumberStyles.Integer, CultureInfo.InvariantCulture);
		}

		private static long ArgL(Dictionary<string, string> a, string key)
		{
			return long.Parse(Arg(a, key), NumberStyles.Integer, CultureInfo.InvariantCulture);
		}

		private static class J
		{
			public static string Str(string s)
			{
				if (s == null)
					return "null";
				StringBuilder sb = new StringBuilder("\"");
				foreach (char c in s)
				{
					switch (c)
					{
						case '"': sb.Append("\\\""); break;
						case '\\': sb.Append("\\\\"); break;
						case '\n': sb.Append("\\n"); break;
						case '\r': sb.Append("\\r"); break;
						case '\t': sb.Append("\\t"); break;
						default:
							if (c < 0x20)
								sb.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
							else
								sb.Append(c);
							break;
					}
				}
				return sb.Append('"').ToString();
			}

			public static string Num(double d)
			{
				if (double.IsNaN(d) || double.IsInfinity(d))
					return "null";
				return d.ToString("R", CultureInfo.InvariantCulture);
			}

			public static string Int(long v)
			{
				return v.ToString(CultureInfo.InvariantCulture);
			}

			public static string Bool(bool b)
			{
				return b ? "true" : "false";
			}

			public static string Obj(params string[] kv)
			{
				StringBuilder sb = new StringBuilder("{");
				for (int i = 0; i + 1 < kv.Length; i += 2)
				{
					if (i > 0)
						sb.Append(',');
					sb.Append(Str(kv[i])).Append(':').Append(kv[i + 1]);
				}
				return sb.Append('}').ToString();
			}

			public static string Arr(IEnumerable<string> items)
			{
				return "[" + string.Join(",", items) + "]";
			}
		}
	}
}
