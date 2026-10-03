Data Card

## 1. Dataset Name

**VGC Battle Logs — Pokémon VGC Online Battles**


---

## 2. Dataset Source

The dataset is a JSON table containing battle data from online Pokémon VGC matches played on **Pokémon Showdown**.

- **Dataset source:** Cameron Angliss / Hugging Face
    
- **Dataset:** `cameronangliss/vgc-battle-logs`
    
- **Original battle platform:** Pokémon Showdown
    
- **Number of battles:** 88,905 scraped battles
    
- **Source reference:** Angliss (2025)
    

We plan to use battles from the **M-A & M-B regulation formats.** These regulations determine which Pokémon, items, and moves are legal in the corresponding competitive format.

---

## 3. License / Permission

The dataset is hosted on Hugging Face, and has the license of MIT.

The producer of the dataset, it allowing us to freely use the documentation files.

---
## 4. Unit of Analysis

The unit of analysis is an **individual Pokémon VGC battle/match**.

Each battle will contain information about each team and their battle. Our project uses the pre-battle information, specifically the team composition and which Pokémon is leading. Based off that information we will attempt o predict the winner using a percentage chance.

---

## 5. Purpose of the Dataset

The dataset contains competitive Pokémon VGC battle records. Our project will use these records to investigate whether information available **before a battle begins** can be used to predict which team will win.

The main  question we are trying to answer is:

> Given two opposing team compositions and their leading Pokémon, can we determine which team is more likely to win before any move is played?

Our project specifically focuses on the predictive value of:

- Team composition
- Starting-lead Pokémon selections

---

## 6. Dataset Pre-process

For the pre-process of our data we used specific regex functions due to how the dataset was structed through Pokemon Showdown. We removed all un-needed information about the whole battle and only grabbed the Pokemon being brought in as well as the two starting leads.

## 7. Dataset Filtering & Cleaning

For filtering and cleaning while looking for empty/junk/error entries, no entries came up therefore not much had to be done in reegards to cleaning and filtering.

## 8. Dataset Size

In total we scraped about **88,905 total battles** from the Pokémon Showdown replay database. We stored them as **JSON tables** in two separate files one for M-A and another for M-B.

---

## 9. Features / Inputs

The main input information planned for the model is:

### Team Composition

Each player has a team of **6 Pokémon**, although only **4 Pokémon can be brought to a match**.

Team composition will provide information about which Pokémon are available on each side of the matchup.

### Starting Leads

Each player will choose 2 Pokémon to lead with, these 2 Pokémon are the main focus for our prediction. The reason this is the main focus is due to how important your lead Pokemon are when participating in a 2v2 style battle.

### Other Potential Information

We also might use things such as, nature, or the held item of the Pokémon to determine the change of winning.

---

## 10. Target Variable

The target is to **produce a percentage value of what team will win the battle**.

This is therefore a:

**Supervised binary classification problem.**

---

## 11. Regulation / Scope

The dataset contains battles from the following Pokémon Champions regulations:

- **M-A**
    
- **M-B**
    

Our project will therefore only consist of matchups under those regulations. The project does not assume that every Pokémon, item, or move is currently available in Pokémon VGC is represented in the dataset. We will however assume that the data does not include any battles, Pokémon, moves, or items from regulation M-C.

---

## 12. Missing Data

This will be investigated during the initial **Exploratory Data Analysis (EDA)**.

information needed:

- Which fields contain missing values
    
- The number/percentage of missing values
    
- Whether missing values are meaningful or represent incomplete battle records
    
- How missing values will be handled during preprocessing
    



---

## 13. Class Balance / Outcome Distribution

This should be calculated during the initial EDA.

Needed information:

- Number of wins for each class
    
- Percentage of battles in each class
    
- Whether the target classes are reasonably balanced
    
- Whether class imbalance needs to be considered when training or evaluating models
    


---

## 14. Sample Inputs and Outputs

### Example Input

A model input will represent information available before a battle begins, such as:

- Team A's composition
   
- Team B's composition
    
- Team A's two starting leads
    
- Team B's two starting leads
    

### Example Output

The model will output a predicted battle outcome, such as:

- Percentage value for **Team A winning**
    
- Percentage value for **Team B winning**
    

**NEEDED INFORMATION:** Output and Input from EDA