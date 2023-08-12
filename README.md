# ⚽️ FPL-Moneyball

## ✍️ Description 
- An attempt to optimise team construction and player transfers in fantasy premier league (FPL) through data analysis and predictive modelling. 
- Model takes a player's rolling average FPL points from previous 5 games and forecasts his upcoming 5-game performance in terms of FPL points. 

## 📊 Data Source 
- Github repo: [Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League)

## 📈 Modelling 
### Baseline 
- Taking a player's mean FPL points from the previous 5 games as an indicator for his next 5-game performance.
- Baseline statistics:
    - rmse: 1.3959271055531288
    - r2: 0.281130180673134

![image](https://github.com/tz1211/FPL-Moneyball/assets/114442618/44dca011-9ce2-42a6-860e-d481a6c37066)

### ML Model 
#### Training Data
- **Inputs** (as rolling average from previous 38 games):
    - Player statistics that directly affect [FPL point allocation](https://www.premierleague.com/news/2174909) (e.g. goals, assists, minutes played, penalties, clean sheets, etc.)
    - Advanced statistics (e.g. xG, xA, etc.)
    - Schedule strengh
- **Labels**
    - Average FPL points over the 38-game period
