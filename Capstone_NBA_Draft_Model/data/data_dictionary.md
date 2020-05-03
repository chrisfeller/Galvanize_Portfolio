# Data Dictionary

## Meta Data

| Column  | Data Type  | Description  |
|---|---|---|
| first_name  | object  | Player's first name  |
| last_name  | object  | Player's last name  |
| bball_ref_name  | object  | Player's alias in Basketball-Reference.com database. Also, corresponds to url of player's individual page on Basketball-Reference.com (i.e. 'karl-anthony-towns-1' in 'https://www.sports-reference.com/cbb/players/karl-anthony-towns-1.html')  |
| sports_ref_name  |  object | Player's alias in Sports-Reference.com database. Also, corresponds to url of player's individual page on Sports-Reference.com (i.e. 'townska01' in  |
| last_college_season  | object  | Player's most recent college season (i.e. 2016-17)  |
| school  | object  | University that player attended  |
| conference  |  object | Conference that player's college competed in  |
| birthday  |  object | Birthdate of player in YYYY-MM-DD format  |
| age_at_draft  | float  | Age in years of player on day of draft  |
| height  | int  | Height of player in inches at time of draft  |
| weight  | int  | Weight of player in inches at time of draft  |


## College Per 100 Possession Data

| Column  | Data Type  | Description  |
|---|---|---|
| college_G  | int  | Games Played  |
| college_GS  | int  | Games Started  |
| college_MP  | int  | Total Minutes Played  |
| FG_per100  | float  | Field Goals Per 100 Possessions  |
| FGA_per100  | float  | Field Goal Attempts Per 100 Possessions  |
| FG%_per100  | float  | Field Goal %  |
| 2P_per100  | float  | 2-Point Field Goals Per 100 Possessions  |
| 2PA_per100  | float  | 2-Point Field Goal Attempts Per 100 Possessions  |
| 2PT%_per100  | float  | 2-Point Field Goal Percentage  |
| 3P_per100  | float  | 3-Point Field Goals Per 100 Possessions  |
| 3PA_per100  | float  | 3-Point Field Goal Attempts Per 100 Possessions  |
| 3PT%_per100  | float  | 3-Point Field Goal Percentage  |
| FT_per100  | float  | Free Throws Per 100 Possessions  |
| FTA_per100  | float  | Free Throw Attempts Per 100 Possessions  |
| FT%_per100  | float  | Free Throw Percentage  |
| TRB_per100  | float  | Total Rebounds Per 100 Possessions  |
| AST_per100  | float  | Assists Per 100 Possessions  |
| STL_per100  | float  | Steals Per 100 Possessions  |
| BLK_per100  | float  | Blocks Per 100 Possessions  |
| TOV_per100  | float  | Turnovers Per 100 Possessions  |
| PF_per100  | float  | Personal Fouls Per 100 Possessions  |
| PTS_per100  | float  | Points Per 100 Possessions  |
| ORtg  | float  | Offensive Rating  |
| DRtg  | float  | Defensive Rating  |


## College Advance Data

| Column  | Data Type  | Description  |
|---|---|---|
| college_PER  | float  | Player Efficiency Rating  |
| college_TS%  | float  | True Shooting Percentage: A measure of shooting efficiency that takes into account 2-point field goals, 3-point field goals, and free throws.  |
| college_eFG%  | float  | Effective Field Goal Percentage: This statistic adjusts for the fact that a 3-point field goal is worth one more point than a 2-point field goal.  |
| college_3PAr  | float  | 3-Point Attempt Rate: Percentage of FG Attempts from 3-Point Range  |
| college_FTr  | float  | Free Throw Rate: Number of FT Attempts Per FG Attempt  |
| college_PProd  | int  | Points Produced: An estimate of the player's offensive points produced.  |
| college_ORB%  | float  | Offensive Rebound Percentage: An estimate of the percentage of available offensive rebounds a player grabbed while he was on the floor.  |
| college_DRB%  | float  | Defensive Rebound Percentage: An estimate of the percentage of available defensive rebounds a player grabbed while he was on the floor.  |
| college_TRB%  | float  | Total Rebound Percentage: An estimate of the percentage of available rebounds a player grabbed while he was on the floor.  |
| college_AST%  | float  | Assist Percentage: An estimate of the percentage of teammate field goals a player assisted while he was on the floor.  |
| college_STL%  | float  | Steal Percentage: An estimate of the percentage of opponent possessions that end with a steal by the player while he was on the floor.  |
| college_BLK%  | float  | Block Percentage: An estimate of the percentage of opponent two-point field goal attempts blocked by the player while he was on the floor.  |
| college_TOV%  | float  | Turnover Percentage: An estimate of turnovers per 100 plays.  |
| college_USG%  | float  | Usage Percentage: An estimate of the percentage of team plays used by a player while he was on the floor.  |
| college_OWS  | float  | Offensive Win Shares: An estimate of the number of wins contributed by a player due to his offense.  |
| college_DWS  | float  | Defensive Win Shares: An estimate of the number of wins contributed by a player due to his defense.  |
| college_WS  | float  | Win Shares: An estimate of the number of wins contributed by a player due to his offense and defense.  |
| college_WS_40  | float  | Win Shares Per 40 Minutes: An estimate of the number of wins contributed by a player per 40 minutes (average is approximately .100).  |
| college_OBPM  | float  | Offensive Box Plus-Minus: A box score estimate of the offensive points per 100 possessions a player contributed above a league-average player, translated to an average team.  |
| college_DBPM  | float  | Defensive Box Plus-Minus: A box score estimate of the defensive points per 100 possessions a player contributed above a league-average player, translated to an average team.  |
| college_BPM  | float  |  Box Plus-Minus: A box score estimate of the points per 100 possessions a player contributed above a league-average player, translated to an average team. |

## College Shooting Distribution Data

| Column  | Data Type  | Description  |
|---|---|---|
| %_shots_at_rim  | float  | %Shots at Rim describes the percentage of a player's field goal attempts that are either classified as layups, dunks, or tip-ins in the play-by-play data.  |
| FG%_at_rim  | float  | FG% at Rim is just the field goal percentage on shots at the rim.  |
| %assisted_at_rim  | float  | %Assisted at Rim is the percentage of made shots at the rim that were assisted. This provides some information as to how often a player is receiving passes that set them up to score at the rim compared with how often that player creates his own shot at the rim.  |
| %_shots_2pt_J  | float  | Percentage of a player's field goal attempts that are from 2-pt range, not including layups, dunks, and tip-ins.  |
| FG%_2pt_Jumpers  | float  | Field Goal Percentage on 2-pt attempts, not including layups, dunks, and tip-ins.   |
| %assisted_2pt_J  | float  | Percentage of made shots from 2-pt range, not including layups, dunks, and tip-ins, that were assisted.  |
| %of_shots_3pt  | float  | Percentage of a player's field goal attempts that are from 3-pt range.  |
| 3FG%  | float  | Field Goal Percentage on 3-pt attempts.  |
| %assisted_3s  | float  | Percentage of made shots from 3-pt range that were assisted.  |
| FTA/FGA  | float  | FTA/FGA is the ratio of free throw attempts to field goal attempts. This is a simple way of measuråing the frequency with which a player gets to the line. Numbers around 0.3 are pretty typical.  |
| FT%  | float  | FT% is the player's percentage from the free throw line.  |

## NBA Advance Data

| Column  | Data Type  | Description  |
|---|---|---|
| NBA_BPM  | float  | Box Plus-Minus: A box score estimate of the points per 100 possessions a player contributed above a league-average player, translated to an average team.  |
| NBA_VORP  | float  | Value Over Replacement Player: A box score estimate of the points per 100 TEAM possessions that a player contributed above a replacement-level (-2.0) player, translated to an average team and prorated to an 82-game season.  |
