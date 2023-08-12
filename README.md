# ⚽️ FPL-Moneyball

## ✍️ Description 
- An attempt to optimise team construction and player transfers in fantasy premier league (FPL) through data analysis and predictive modelling. 
- Model takes a player's rolling average FPL points from previous 5 games and forecasts his upcoming 5-game performance in terms of FPL points. 

## 📊 Data Source 
- Github repo: [Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League)

## 📈 Modelling 
### Baseline 
- Taking a player's mean FPL points from the previous 10 games as an indicator for his next 5-game performance.
- Baseline statistics:
    - rmse: 1.256293266288406
    - r2: 0.3992381036273135

![image](https://github.com/tz1211/FPL-Moneyball/assets/114442618/de586100-91e4-4cfa-b1fe-078177f2f11e)


### ML Model 
#### Training Data
- **Inputs** (as rolling average from previous 38 games):
    - Player statistics that directly affect [FPL point allocation](https://www.premierleague.com/news/2174909) (e.g. goals, assists, minutes played, penalties, clean sheets, etc.)
    - Advanced statistics (e.g. xG, xA, etc.)
    - Schedule strengh
- **Labels**
    - Average FPL points over the 38-game period
