## 1. Dataset Name
**VGC Battle Logs — Pokémon VGC Online Battles**


## 2. Dataset Source
The dataset is a JSON table containing battle data scraped from online Pokémon VGC matches played on **Pokémon Showdown**.
- **Dataset source:** Cameron Angliss / Hugging Face
- **Dataset:** `cameronangliss/vgc-battle-logs`
- **Original battle platform:** Pokémon Showdown
- **Number of battles:** 7,229 scraped best-of-1 battles (6,905 Regulation M-A, 324 Regulation M-B)
- **Source reference:** Angliss (2025)

We plan to use battles from the **M-A & M-B regulation formats.** These regulations determine which Pokémon, items, and moves are legal in the corresponding competitive format.


## 3. License / Permission
- **License**: Permissive Open-Access (MIT License)
- **Data Reuse Rights**: Standard academic and research reuse granted by dataset creator.
- **Privacy**: Contains no personal information, human subject records, employment metrics, or sensitive medical attributes.
- **Platform Terms**: Sourced entirely from public, anonymized Pokémon Showdown match replays.


## 4. Unit of Analysis
The unit of analysis is an **individual Pokémon VGC battle/match**.
Each battle will contain information about each team and their battle. Our project uses the pre-battle information, specifically the team composition and which Pokémon is leading. Based off that information we will attempt a predict the winner using a percentage chance.


## 5. Purpose of the Dataset
The dataset contains competitive Pokémon VGC battle records. Our project will use these records to investigate whether information available **before a battle begins** can be used to predict which team will win.

The main question we are trying to answer is:
> Given two opposing team compositions and their leading Pokémon, can we determine which team is more likely to win before any move is played?

Our project specifically focuses on the predictive value of:
- Team composition
- Starting-lead Pokémon selections


## 6. Dataset Pre-process
For the pre-process of our data we used specific regex functions due to how the dataset was structured through Pokémon Showdown. We removed all unnecessary information about the entire battle and only grabbed the Pokémon being brought in as well as the two starting leads.


## 7. Dataset Filtering & Cleaning
For filtering and cleaning while looking for empty/junk/error entries, no entries came up therefore not much had to be done in reegards to cleaning and filtering.


## 8. Dataset Size
In total we imported **7,229 total battles** which originated from the Pokémon Showdown replay database. We stored them as **JSON tables** in two separate files one for M-A and another for M-B. In the EDA (`notebooks/01_milestone2_exploratory_data_analysis.ipynb`) these tables are loaded as Pandas DataFrame objects to help with visualization.

Before data processing / cleaning (as raw imported files), data size for the Regulation M-A file was ~50 MB, and ~2.3 MB for Regulation M-B. Afterwards, sizes were reduced to ~6.5 MB and ~300 KB for Regulations M-A and M-B JSON files respectively.


## 9. Features / Inputs
The main input information planned for the model is:

### Team Composition
Each player has a team of **6 Pokémon**, although only **4 Pokémon can be brought to a match**.
Team composition will provide information about which Pokémon are available on each side of the matchup.

### Starting Leads
Each player will choose 2 Pokémon to lead with, these 2 Pokémon are the main focus for our prediction. The reason this is the main focus is due to how important your lead Pokémon are when participating in a 2v2 style battle.

### Other Potential Information
For leading Pokémon, additional information of nature, held item, moves and ability of each Pokémon will also be factored in to determine matchups.


## 10. Target Variable
The target is to **produce a percentage value of what team will win the battle**.

This is therefore a:
**Supervised binary classification problem.**


## 11. Regulation / Scope
The dataset contains battles from the following Pokémon Champions regulations:

- **M-A**
- **M-B**

Our project will only consist of matchups under these regulations. The project does not assume that every Pokémon, item, or move is currently available in Pokémon VGC is represented in the dataset. We will however assume that the data does not include any battles, Pokémon, moves, or items from other regulations (such as the current running Regulation M-C).


## 12. Missing Data
The Python `missingno` package was utilized to find any missing data throughout the files and overall none was found; however, there is an issue of ~75% of Pokémon nature values in the Regulation M-A file being empty; due to natures not being open information to players as of that point in the regulation. This issue will be dealt with via a feature model that uses context from the rest of the dataset (such as Pokémon species and what items they are holding) to estimate what nature a specified Pokémon may be.


## 13. Class Balance / Outcome Distribution
Our data holds a total of 7229 unique battles. ~95.5% (6905) of these battles are in Regulation M-A, and the remaining ~4.5% (324) are in Regulation M-B. Almost all of our battle data is in Regulation M-A. The outcome distribution is quite evenly balanced, where the total wins for player 1 across both Regulations is 3658, and for player 2 3571; this is almost a 50-50 split. Imbalance is not an issue for training or evaluating, however training and testing will mainly be done with Reg. M-A battles, and validation with M-B.


## 14. Sample Inputs and Outputs

### Example Input
As our models will analyze separate battles at a time, a sample input would consist of one entry from the processed JSON file. This file contains the following data that can be passed to features:

- Each player's:
    - leading Pokémon
    - leading Pokémons' moves
    - leading Pokémons' ability
    - leading Pokémons' held item
    - leading Pokémons' nature
    - back Pokémon (the remaining Pokémon in their team)
 - winner

The winner will not be passed to model features and remains in the data for validation.

### Example Output
Features will process various aspects about the input data and extract information separately. For example, a feature that is concerned with learning the speed control of each player's leads in a specific battle will output a new JSON file containing integer values of related cases, like... 

```
...
'p1_speed_control_score':'1',   # score integer value
'p2_speed_control_score':'0',
'p1_has_speed_advantage':'1'    # binarized Boolean value
...
```

This output data would then be incorporated along with various other output data together to help weigh values of the final model outputs:

- Percentage value for **Player A winning**
- Percentage value for **Player B winning**
