@echo off
REM scripts/train.bat
REM CryptoTrader training launcher for cmd.exe (double-click or: scripts\train.bat)
REM Edit the values below, save, run. Each SET maps to a config field via the
REM CT_<SECTION>__<FIELD> convention (overrides config\config.yaml for this run).

REM Run from the repo root (the folder this script's parent lives in).
cd /d "%~dp0\.."

REM --- Data to train on ---
set DAYS=365
set EXCHANGE=binance
set SYMBOL=BTC/USDT
set TIMEFRAME=15m

REM --- Barriers (labels + exits) ---
set CT_BARRIERS__TP_MULT=2.5
set CT_BARRIERS__SL_MULT=1.0
set CT_BARRIERS__HORIZON=15

REM --- Strategy thresholds (higher = fewer, better trades) ---
set CT_STRATEGY__LONG_THRESHOLD=0.66
set CT_STRATEGY__SHORT_THRESHOLD=0.66

REM --- Risk / cost control ---
set CT_RISK__MIN_EDGE_COST_RATIO=3.0
set CT_RISK__MAX_LEVERAGE=1.0
set CT_RISK__COOLDOWN_BARS=5
set CT_RISK__RISK_PER_TRADE=0.005

REM --- Model (LightGBM) ---
set CT_MODEL__N_ESTIMATORS=800
set CT_MODEL__LEARNING_RATE=0.02
set CT_MODEL__RANDOM_STATE=42
REM set CT_MODEL__CLASS_WEIGHT=balanced   (uncomment to A/B test balanced weighting)

echo Training on %EXCHANGE% %SYMBOL% %TIMEFRAME%, %DAYS% days...
python scripts\train_model.py --days %DAYS% --exchange %EXCHANGE% --symbol %SYMBOL% --timeframe %TIMEFRAME%

pause
