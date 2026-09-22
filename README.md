# TickerArc

**Stock Analysis and Prediction System**

TickerArc combines market data, technical analysis, deep learning and an explicit analysis-agent layer to study future stock movement.

## Core capabilities

- NIFTY 50 historical data ingestion
- Pandas + TA-Lib feature engineering
- Full TA-Lib CDL candlestick feature family
- Heuristic chart-structure detection
- Multi-horizon LSTM returns for 1D / 5D / 10D
- Direction probabilities and 5-day volatility output
- Activity, low-volume and low-attention scanners
- Heuristic 0–100 potential score
- LSTM-DQN reinforcement-learning policy
- 24-year training / 2-year chronological holdout evaluation
- Expanding walk-forward evaluation
- Black-and-white Streamlit dashboard
- 1-minute or 3-minute live/near-live refresh
- Single entry point: `app.py`

## Run the complete application

Clone the repository and install dependencies:

```bash
git clone https://github.com/Riyasantro/tickerarc-stock-prediction.git
cd tickerarc-stock-prediction

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start TickerArc:

```bash
streamlit run app.py
```

On first launch, the application:

1. Downloads maximum available NIFTY 50 daily history.
2. Builds the TA-Lib and chart-pattern feature dataset.
3. Uses a chronological 24-year training window and reserves the latest 2 years as a holdout set.
4. Trains the global multi-stock LSTM without using holdout targets.
5. Evaluates the frozen model on the 2-year holdout.
6. Loads the model and starts the dashboard.

After initialization, the dashboard fetches the live/near-live market layer on the selected 1-minute or 3-minute interval and runs inference without retraining the model.

## Dashboard

The app shows:

- High activity stocks
- Low activity stocks
- Low relative-volume stocks
- Low-attention / popularity-proxy stocks
- Current price and session change
- 1D / 5D / 10D modeled returns
- Up / neutral / down probabilities
- 5D volatility estimate
- Potential score
- Candlestick pattern detections
- Chart-pattern detections
- Support and resistance overlays
- Analysis-agent summary
- On-demand expanding walk-forward evaluation

## Modeling

The production prediction model is a multi-task LSTM:

```text
60-day feature sequence
        |
      LSTM
        |
   Shared latent
   /     |      \
returns direction volatility
  |        |
1D/5D/10D  3-class probability
```

The reinforcement-learning component is an LSTM-DQN policy with Sell/Hold/Buy actions. It is kept separate from the supervised return forecaster so the UI can show both the numerical forecast and the learned policy.

The potential score is a heuristic aggregation of model probability, expected return, activity, volume and detected pattern signals. It is not a guaranteed price target.

## Live data note

The app uses a free public market-data layer. Refreshing the Streamlit interface every minute does not guarantee a new exchange tick every minute; the dashboard reports the provider timestamp and falls back between 1-minute and 5-minute intraday data when necessary.

## Evaluation

The selected-stock panel includes an on-demand expanding walk-forward evaluation. Each fold trains on earlier observations and is scored on later observations only, preserving time order.

The primary evaluation reports:

- 5-day return MAE
- Direction accuracy
- Number of 2-year out-of-sample samples

The live dashboard then runs inference from the latest completed daily feature sequence using the trained LSTM. The 2-year holdout is never used as a training label.

Model checkpoints and downloaded market data are intentionally ignored by Git.


### History availability note

The 24-year training / 2-year holdout split is applied stock-by-stock using each symbol's maximum available history. Stocks that do not have enough history for the full window are excluded from the exact holdout training set, while the dashboard can still display their available market data.


### Persistent online reinforcement loop

During a live Streamlit session, the selected stock has an independent online RL feedback loop:

```text
Current interval
    ↓
LSTM-DQN chooses Sell / Hold / Buy
    ↓
wait 1 or 3 minutes
    ↓
observe realized price move
    ↓
calculate reward
    ↓
Replay Buffer + DQN update
    ↓
persist checkpoint + replay memory
```

The online policy is separate from the supervised 24-year LSTM forecaster. The supervised LSTM produces the numerical 1D / 5D / 10D forecast, while LSTM-DQN learns a trading policy from realized interval rewards.

The online RL checkpoint and replay memory are stored locally under `models/` and are ignored by Git.

When the application is restarted, TickerArc restores the online policy, optimizer state, replay memory and the selected-stock runtime state. It then replays available intraday history since the last saved decision before returning to the current live interval. On a first online-RL launch, the replay window starts with the most recent two calendar days.

The catch-up window depends on the intraday history available from the provider. yfinance documents intraday history separately from daily history and notes that intraday requests cannot extend beyond the provider's available recent window. citeturn245173search0turn245173search1
