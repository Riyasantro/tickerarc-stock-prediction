# TickerArc

**Stock Analysis and Prediction System**

TickerArc combines market data, technical analysis, deep learning and an explicit analysis-agent layer to study future stock movement.

## Core capabilities

- NIFTY 50 historical data ingestion
- Pandas + TA-Lib feature engineering
- Full TA-Lib CDL candlestick feature family
- Heuristic chart-structure detection
- Multi-horizon LSTM returns for 1D / 5D / 10D
- Temporal-attention pooling over the LSTM sequence
- Causal trend, volatility, drawdown and volume-pressure regime features
- Direction probabilities and 5-day volatility output
- Activity, low-volume and low-attention scanners
- Heuristic 0–100 potential score
- LSTM-DQN reinforcement-learning policy
- Persistent online RL feedback and restart/catch-up learning
- 20-year fit + 4-year validation + 2-year untouched chronological test
- Final model refit on the complete 24-year development window after validation-based epoch selection
- Baseline benchmarking against zero-return and Ridge models on the same 2-year test
- Expanding walk-forward evaluation
- Dark-green professional trading-terminal Streamlit UI with interactive Plotly charts
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

After initialization, the dashboard fetches the live/near-live market layer on the selected 1-minute or 3-minute interval. The supervised forecast model stays frozen; the separate online LSTM-DQN policy learns from newly realized interval rewards.

## Dashboard

The app shows:

- Most active stocks
- Popular / attention-proxy stocks
- Model buy-signal stocks
- Potential leaders
- Low-attention watchlist shown as a separate category
- Dedicated instrument workspace for each selected NIFTY 50 stock
- Current price and session change
- 1D / 5D / 10D modeled returns
- Up / neutral / down probabilities
- 5D volatility estimate
- Potential score
- Candlestick pattern detections
- Chart-pattern detections
- Support/resistance overlays, EMA 20/50, volume and RSI panels
- Interactive zoom/scroll trading chart with selectable 1M/3M/6M/1Y/3Y/5Y/MAX ranges
- Model probability distribution chart
- Analysis-agent summary
- Model performance charts against the zero-return and Ridge baselines
- On-demand expanding walk-forward evaluation

## Modeling

The production prediction model is a multi-task LSTM with temporal attention pooling:

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

### Model selection and test protocol

```text
maximum history
      |
      +--> 20Y fit
      |       |
      |   validation
      |      4Y
      |       |
      +-------+
              |
       select training duration
              |
       refit on complete 24Y
              |
       frozen final model
              |
        latest 2Y test
```

The validation window is used only for training-duration selection. The latest 2 years remain untouched until final evaluation. Baselines use the same final test window so the LSTM result has a reference point.

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

The online policy is separate from the supervised 24-year LSTM forecaster. The supervised LSTM produces the numerical 1D / 5D / 10D forecast, while LSTM-DQN learns a trading policy from realized interval rewards. The offline RL training/evaluation scripts also use a 24-year training window and latest 2-year holdout.

The online RL checkpoint and replay memory are stored locally under `models/` and are ignored by Git.

When the application is restarted, TickerArc restores the online policy, optimizer state, replay memory and the selected-stock runtime state. It then replays available intraday history since the last saved decision before returning to the current live interval. On a first online-RL launch, the replay window starts with the most recent two calendar days.

The catch-up window depends on the intraday history available from the provider. The online loop never fabricates missing intervals; it replays only the intraday bars that can actually be fetched.


## Streamlit Community Cloud deployment

TickerArc can be deployed from the private GitHub repository through Streamlit Community Cloud. Community Cloud supports private repositories when the GitHub connection is granted the additional repository access. The deployment entrypoint is `app.py`. The current dependency pins use Streamlit 1.64.0 and TA-Lib 0.6.8.

For the first deployment, choose Python 3.12 in Streamlit's Advanced settings so the local and cloud environments match. Community Cloud currently defaults to Python 3.12 and supports other maintained Python versions.

The current full application performs substantial startup work: maximum-history download, feature construction, 20Y fit + 4Y validation, 24Y refit, 2Y holdout evaluation, and model loading. Community Cloud currently provides approximately 2 CPU cores maximum and 2.7 GB memory, so this bootstrap can be too heavy for a reliable cold start.

Also note that files generated while an app is running on Community Cloud are not guaranteed to persist across user sessions. Therefore the local online-RL checkpoint/replay files are suitable for local development but are not a reliable persistence layer for the deployed app. For production-style persistent online RL, store checkpoints/replay/runtime state in an external persistent database or object store and load them on startup.

Community Cloud apps without traffic hibernate after 12 hours, so the 1/3-minute online learning loop should not be treated as a continuously running background worker on the free service.

## Professional trading terminal layout

```text
Market header / live breadth
        |
Most Active   | Popular / Attention
Model Signals | Potential Leaders
Low Attention Watch
        |
Selected Instrument
        |
Candles + EMA20/EMA50 + Support/Resistance
Volume panel
RSI panel
        |
Overview | Signals | Model Performance | Online RL
```

The Streamlit interface is organized as a market workspace rather than a raw data table.
The sidebar is reserved for workspace controls, live market findings and system health. It does not repeat basic model descriptions.

## Production deployment boundary

The application code is structured for production-style operation: cached market reads, frozen supervised inference after model creation, explicit holdout evaluation, isolated online RL state, provider timestamps, health indicators and failure handling.

The remaining production infrastructure requirement is persistent external storage for online-RL checkpoints/replay/runtime state when deployed on an ephemeral hosting platform. The application should not assume that local files inside such a service are durable across restarts. A persistent object store or database is required for durable cloud RL state.
