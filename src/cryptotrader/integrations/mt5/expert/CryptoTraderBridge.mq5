//+------------------------------------------------------------------+
//|                                        CryptoTraderBridge.mq5     |
//|   Bridges a MetaTrader 5 terminal to the CryptoTrader dashboard.  |
//|                                                                   |
//|   Every PollSeconds the EA sends its recent CLOSED bars + current |
//|   position + account to  <ServerURL>/api/mt5/decide  and executes |
//|   the returned decision (open/close/hold) on the broker, using    |
//|   the server's protective stop-loss / take-profit and a risk-     |
//|   based lot size. The ML/decision logic stays on the server.      |
//|                                                                   |
//|   SETUP (once):  Tools > Options > Expert Advisors >              |
//|     "Allow WebRequest for listed URL" and add your ServerURL      |
//|     (e.g. https://luciphy.com). Then attach this EA to a chart    |
//|     whose symbol+timeframe match a trained model.                 |
//+------------------------------------------------------------------+
#property copyright "CryptoTrader"
#property version   "1.00"
#property strict

#include <Trade/Trade.mqh>

input string  ServerURL        = "https://luciphy.com"; // dashboard base URL (must be whitelisted)
input string  ApiToken         = "";                    // mt5.api_token (X-API-Token header)
input int     BarsToSend       = 320;                   // closed bars per request (>= model warmup)
input int     PollSeconds      = 60;                    // how often to ask for a decision
input int     RequestTimeoutMs = 8000;                  // WebRequest timeout
input long    MagicNumber      = 770011;                // identifies this EA's positions
input double  MaxSpreadPoints  = 0;                     // skip trading if spread > this (0 = off)
input bool    AllowShort       = true;                  // permit short entries
input bool    UseServerSLTP    = true;                  // attach server SL/TP to orders
input double  FallbackRiskPct  = 0.01;                  // risk fraction if server omits one
input double  MaxLots          = 0;                     // hard lot cap (0 = broker max)

CTrade  trade;

//+------------------------------------------------------------------+
int OnInit()
  {
   trade.SetExpertMagicNumber(MagicNumber);
   trade.SetTypeFillingBySymbol(_Symbol);
   if(ApiToken=="")
      Print("WARNING: ApiToken is empty — set it to mt5.api_token from the server config.");
   if(!IsUrlAllowed())
      Print("ACTION REQUIRED: add ", ServerURL,
            " under Tools>Options>Expert Advisors>Allow WebRequest, then re-attach.");
   EventSetTimer(MathMax(5, PollSeconds));
   Print("CryptoTraderBridge started on ", _Symbol, " ", PeriodToTf(),
         " -> ", ServerURL, "/api/mt5/decide");
   Cycle();   // act immediately on attach
   return(INIT_SUCCEEDED);
  }

void OnDeinit(const int reason) { EventKillTimer(); }
void OnTimer()                  { Cycle(); }

//+------------------------------------------------------------------+
//| One decide->execute cycle                                        |
//+------------------------------------------------------------------+
void Cycle()
  {
   string body  = BuildRequest();
   if(body=="") return;
   string resp  = "";
   int    code  = HttpPost(ServerURL+"/api/mt5/decide", body, resp);
   if(code!=200)
     {
      PrintFormat("decide HTTP %d: %s", code, StringSubstr(resp,0,200));
      return;
     }

   string action = JsonStr(resp,"action");
   string reason = JsonStr(resp,"reason");
   PrintFormat("decision: action=%s dir=%s conf=%.3f reason=%s",
               action, JsonStr(resp,"direction"), JsonDbl(resp,"confidence"), reason);

   if(action=="open_long")       OpenTrade(true,  resp);
   else if(action=="open_short") { if(AllowShort) OpenTrade(false, resp); }
   else if(action=="close")      CloseOwn();
   // "hold" / "none" -> do nothing
  }

//+------------------------------------------------------------------+
//| Build the JSON request: bars + position + account                |
//+------------------------------------------------------------------+
string BuildRequest()
  {
   MqlRates rates[];
   ArraySetAsSeries(rates,true);
   int want = BarsToSend+1;                       // +1 because index 0 is the forming bar
   int got  = CopyRates(_Symbol,_Period,0,want,rates);
   if(got<30) { Print("not enough bars yet (",got,")"); return ""; }

   string bars = "";
   // Oldest -> newest, skipping the still-forming current bar (index 0).
   for(int i=got-1; i>=1; i--)
     {
      if(bars!="") bars += ",";
      bars += StringFormat("{\"t\":%I64d,\"o\":%.8g,\"h\":%.8g,\"l\":%.8g,\"c\":%.8g,\"v\":%.1f}",
                           (long)rates[i].time, rates[i].open, rates[i].high,
                           rates[i].low, rates[i].close, (double)rates[i].tick_volume);
     }

   string pos = PositionJson();
   string acc = StringFormat("{\"equity\":%.2f,\"balance\":%.2f,\"currency\":\"%s\"}",
                             AccountInfoDouble(ACCOUNT_EQUITY),
                             AccountInfoDouble(ACCOUNT_BALANCE),
                             AccountInfoString(ACCOUNT_CURRENCY));

   return StringFormat("{\"symbol\":\"%s\",\"timeframe\":\"%s\",\"position\":%s,\"account\":%s,\"bars\":[%s]}",
                       _Symbol, PeriodToTf(), pos, acc, bars);
  }

//+------------------------------------------------------------------+
//| Current position (this EA's magic) as JSON, or null              |
//+------------------------------------------------------------------+
string PositionJson()
  {
   if(!PositionSelect(_Symbol)) return "null";
   if(PositionGetInteger(POSITION_MAGIC)!=MagicNumber) return "null";
   string side = (PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY) ? "long" : "short";
   datetime opened = (datetime)PositionGetInteger(POSITION_TIME);
   int held = iBarShift(_Symbol,_Period,opened,false);    // bars since entry
   return StringFormat("{\"side\":\"%s\",\"volume\":%.4f,\"entry_price\":%.8g,\"bars_held\":%d}",
                       side, PositionGetDouble(POSITION_VOLUME),
                       PositionGetDouble(POSITION_PRICE_OPEN), held);
  }

//+------------------------------------------------------------------+
//| Open a position sized from the server's risk fraction + stop     |
//+------------------------------------------------------------------+
void OpenTrade(bool isLong, string resp)
  {
   if(HasOwnPosition()) return;                    // already in a managed position
   if(SpreadTooWide())  { Print("spread too wide; skip entry"); return; }

   double sl = UseServerSLTP ? JsonDbl(resp,"stop_loss")    : 0.0;
   double tp = UseServerSLTP ? JsonDbl(resp,"take_profit")  : 0.0;
   double stopDist = JsonDbl(resp,"stop_distance");
   double riskFrac = JsonDbl(resp,"risk_fraction");
   if(riskFrac<=0) riskFrac = FallbackRiskPct;

   double lots = LotsForRisk(riskFrac, stopDist);
   if(lots<=0) { Print("computed lot size <= 0; skip"); return; }

   sl = NormalizeStops(sl, isLong, true);
   tp = NormalizeStops(tp, isLong, false);

   bool ok = isLong ? trade.Buy(lots,_Symbol,0.0,sl,tp,"CryptoTrader")
                    : trade.Sell(lots,_Symbol,0.0,sl,tp,"CryptoTrader");
   if(!ok)
      PrintFormat("order failed: %d %s", trade.ResultRetcode(), trade.ResultRetcodeDescription());
   else
      PrintFormat("%s %.4f lots  SL=%.8g TP=%.8g", isLong?"BUY":"SELL", lots, sl, tp);
  }

//+------------------------------------------------------------------+
//| Risk-based position size: risk = balance*frac, loss at stopDist  |
//+------------------------------------------------------------------+
double LotsForRisk(double riskFrac, double stopDist)
  {
   double tickValue = SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_VALUE);
   double tickSize  = SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_SIZE);
   double minLot    = SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MIN);
   double maxLot    = SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MAX);
   double lotStep   = SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_STEP);
   if(tickSize<=0 || tickValue<=0 || stopDist<=0) return minLot;

   double riskMoney = AccountInfoDouble(ACCOUNT_BALANCE)*riskFrac;
   double lossPerLot = (stopDist/tickSize)*tickValue;     // money lost per 1.0 lot at the stop
   if(lossPerLot<=0) return minLot;

   double lots = riskMoney/lossPerLot;
   lots = MathFloor(lots/lotStep)*lotStep;                // round DOWN to a valid step
   if(MaxLots>0) maxLot = MathMin(maxLot, MaxLots);
   lots = MathMax(minLot, MathMin(lots, maxLot));
   return lots;
  }

//+------------------------------------------------------------------+
//| Clamp SL/TP to the broker's minimum stop distance                |
//+------------------------------------------------------------------+
double NormalizeStops(double price, bool isLong, bool isStop)
  {
   if(price<=0) return 0.0;
   int    digits = (int)SymbolInfoInteger(_Symbol,SYMBOL_DIGITS);
   double point  = SymbolInfoDouble(_Symbol,SYMBOL_POINT);
   long   stopsL = SymbolInfoInteger(_Symbol,SYMBOL_TRADE_STOPS_LEVEL);
   double minDist= stopsL*point;
   double ref    = isLong ? SymbolInfoDouble(_Symbol,SYMBOL_ASK)
                          : SymbolInfoDouble(_Symbol,SYMBOL_BID);
   // Ensure the level sits on the correct side and beyond the broker minimum.
   if(isStop)
     {
      if(isLong  && price > ref-minDist) price = ref-minDist;
      if(!isLong && price < ref+minDist) price = ref+minDist;
     }
   else
     {
      if(isLong  && price < ref+minDist) price = ref+minDist;
      if(!isLong && price > ref-minDist) price = ref-minDist;
     }
   return NormalizeDouble(price,digits);
  }

//+------------------------------------------------------------------+
void CloseOwn()
  {
   if(!HasOwnPosition()) return;
   if(!trade.PositionClose(_Symbol))
      PrintFormat("close failed: %d %s", trade.ResultRetcode(), trade.ResultRetcodeDescription());
   else
      Print("position closed");
  }

bool HasOwnPosition()
  {
   if(!PositionSelect(_Symbol)) return false;
   return (PositionGetInteger(POSITION_MAGIC)==MagicNumber);
  }

bool SpreadTooWide()
  {
   if(MaxSpreadPoints<=0) return false;
   double sp = (SymbolInfoDouble(_Symbol,SYMBOL_ASK)-SymbolInfoDouble(_Symbol,SYMBOL_BID))
               / SymbolInfoDouble(_Symbol,SYMBOL_POINT);
   return (sp > MaxSpreadPoints);
  }

//+------------------------------------------------------------------+
//| HTTP POST JSON with the token header. Returns the HTTP status.   |
//+------------------------------------------------------------------+
int HttpPost(string url, string body, string &response)
  {
   char post[]; char result[]; string rheaders;
   int len = StringToCharArray(body, post, 0, WHOLE_ARRAY, CP_UTF8)-1;  // drop the null terminator
   if(len<0) len=0;
   ArrayResize(post,len);
   string headers = "Content-Type: application/json\r\nX-API-Token: "+ApiToken+"\r\n";
   ResetLastError();
   int code = WebRequest("POST", url, headers, RequestTimeoutMs, post, result, rheaders);
   if(code==-1)
     {
      PrintFormat("WebRequest error %d (is %s whitelisted?)", GetLastError(), url);
      return -1;
     }
   response = CharArrayToString(result, 0, WHOLE_ARRAY, CP_UTF8);
   return code;
  }

//+------------------------------------------------------------------+
//| Minimal flat-JSON readers (no nested objects in our response)    |
//+------------------------------------------------------------------+
string JsonStr(string js, string key)
  {
   string pat = "\""+key+"\"";
   int p = StringFind(js,pat);
   if(p<0) return "";
   p = StringFind(js,":",p+StringLen(pat));
   if(p<0) return "";
   int i=p+1, n=StringLen(js);
   while(i<n && StringGetCharacter(js,i)==' ') i++;
   if(i<n && StringGetCharacter(js,i)=='\"')
     {
      int e=StringFind(js,"\"",i+1);
      if(e<0) return "";
      return StringSubstr(js,i+1,e-(i+1));
     }
   int e=i;
   while(e<n)
     {
      ushort ch=StringGetCharacter(js,e);
      if(ch==','||ch=='}'||ch==' ') break;
      e++;
     }
   return StringSubstr(js,i,e-i);
  }

double JsonDbl(string js, string key)
  {
   string v = JsonStr(js,key);
   if(v=="" || v=="null") return 0.0;
   return StringToDouble(v);
  }

//+------------------------------------------------------------------+
//| _Period -> the server's timeframe string                         |
//+------------------------------------------------------------------+
string PeriodToTf()              { return PeriodToTfOf(_Period); }
string PeriodToTfOf(ENUM_TIMEFRAMES tf)
  {
   switch(tf)
     {
      case PERIOD_M1:  return "1m";
      case PERIOD_M5:  return "5m";
      case PERIOD_M15: return "15m";
      case PERIOD_M30: return "30m";
      case PERIOD_H1:  return "1h";
      case PERIOD_H4:  return "4h";
      case PERIOD_D1:  return "1d";
      case PERIOD_W1:  return "1w";
      default:         return "1h";
     }
  }

//+------------------------------------------------------------------+
bool IsUrlAllowed()
  {
   // No direct API to read the whitelist; a probe GET tells us if WebRequest is permitted.
   char post[]; char result[]; string rh;
   ResetLastError();
   int code = WebRequest("GET", ServerURL+"/api/health", "", 3000, post, result, rh);
   return (code!=-1);
  }
//+------------------------------------------------------------------+
