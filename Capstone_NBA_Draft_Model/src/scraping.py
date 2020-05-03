import requests
import numpy as np
import pandas as pd
from time import sleep
from bs4 import BeautifulSoup as BS
from cleaning import player_names

def draft_picks():
    """
    Scrape names and supplemental information of all players drafted to the NBA
    since 2011 from Basketball-Reference.com's Draft Finder.

    Args:
        None

    Returns:
        draft_picks: pandas DataFrame with all players drafted to the NBA since
        2011. DataFrame includes player's name, age, draft selection, position,
        and other supplemental information.
    """
    draft_picks = pd.DataFrame()
    for i in range(0, 500, 100):
        url = "https://www.basketball-reference.com/play-index/draft_finder.cgi?request=1&year_min=2011&year_max=2017&round_min=&round_max=&pick_overall_min=&pick_overall_max=&franch_id=&college_id=0&is_active=&is_hof=&pos_is_g=Y&pos_is_gf=Y&pos_is_f=Y&pos_is_fg=Y&pos_is_fc=Y&pos_is_c=Y&pos_is_cf=Y&c1stat=&c1comp=&c1val=&c2stat=&c2comp=&c2val=&c3stat=&c3comp=&c3val=&c4stat=&c4comp=&c4val=&order_by=year_id&order_by_asc=&offset={}#stats::none".format(i)
        table = pd.read_html(url)[0]
        draft_picks = pd.concat([draft_picks, table])
    draft_picks.columns = draft_picks.columns.droplevel()
    mask = (draft_picks['Player'].notnull()) & (draft_picks['Player'] != 'Player')
    draft_picks = draft_picks[mask]
    return draft_picks

def college_per100(college_names):
    """
    Scrape collegiate Per 100 Possessions data for all players that played in
    college and were drafted into the NBA since 2011. Per 100 Possessions data
    comes from Sports-Reference.com. For each player, their most recent season
    is scraped.

    Args:
        college_names: pandas DataFrame that inclues player's name and
        Sports-Reference.com alias for use in url string formatting.

    Returns:
        per100: pandas DataFrame with Per 100 Possessions data from the most
        recent college season for each player.
    """
    per100 = pd.DataFrame()
    for index, row in college_names.iterrows():
        sleep(np.random.normal(3))
        url = "https://www.sports-reference.com/cbb/players/{0}.html#players_per_poss::none".format(row['sports_ref_name'])
        html = requests.get(url).text
        soup = BS(html, 'html.parser')
        placeholders = soup.find_all('div', {'class': 'placeholder'})
        for x in placeholders:
            comment = ''.join(x.next_siblings)
            soup_comment = BS(comment, 'html.parser')
            tables = soup_comment.find_all('table', attrs={"id":"players_per_poss"})
            for tag in tables:
                df = pd.read_html(tag.prettify())[0]
                table = df.iloc[-2, :]
                per100 = per100.append(table).reset_index()
                per100 = per100[['Season', 'School', 'Conf', 'G', 'GS', 'MP',
                                'FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P',
                                '3PA', '3P%', 'FT', 'FTA', 'FT%', 'TRB', 'AST',
                                'STL', 'BLK', 'TOV', 'PF', 'PTS', 'ORtg', 'DRtg']]
    return per100

def college_advance(college_names):
    """
    Scrape collegiate Advance data for all players that played in
    college and were drafted into the NBA since 2011. Advance data
    comes from Sports-Reference.com. For each player, their most recent season
    is scraped.

    Args:
        college_names: pandas DataFrame that inclues player's name and
        Sports-Reference.com alias for use in url string formatting.

    Returns:
        advance: pandas DataFrame with Advance data from the most
        recent college season for each player.
    """
    advance = pd.DataFrame()
    for index, row in college_names.iterrows():
        sleep(np.random.normal(3))
        url = "https://www.sports-reference.com/cbb/players/{0}.html#players_advanced::none".format(row['sports_ref_name'])
        html = requests.get(url).text
        soup = BS(html, 'html.parser')
        placeholders = soup.find_all('div', {'class': 'placeholder'})
        for x in placeholders:
            comment = ''.join(x.next_siblings)
            soup_comment = BS(comment, 'html.parser')
            tables = soup_comment.find_all('table', attrs={"id":"players_advanced"})
            for tag in tables:
                df = pd.read_html(tag.prettify())[0]
                table = df.iloc[-2, :]
                advance = advance.append(table).reset_index()
                advance = advance[['Season', 'School', 'Conf', 'G', 'GS', 'MP',
                                    'PER', 'TS%', 'eFG%', '3PAr', 'FTr', 'PProd',
                                    'ORB%', 'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%',
                                    'TOV%', 'USG%', 'OWS', 'DWS', 'WS', 'WS/40',
                                    'OBPM', 'DBPM', 'BPM']]
    return advance

def nba_advance(college_names):
    """
    Scrape NBA Advance data for all players that played in
    college and were drafted into the NBA since 2011. Advance data
    comes from Basketball-Reference.com. For each player, their second NBA-season
    is scraped.

    Args:
        college_names: pandas DataFrame that inclues player's name and
        Basketball-Reference.com alias for use in url string formatting.

    Returns:
        advance: pandas DataFrame with Advance data from the second NBA season
        for each player.
    """
    advance = pd.DataFrame()
    for index, row in college_names.iterrows():
        sleep(np.random.normal(3))
        url = 'https://www.basketball-reference.com/players/{0}/{1}.html#advanced::none'.format(row['bball_ref_name'][0], row['bball_ref_name'])
        html = requests.get(url).text
        soup = BS(html, 'html.parser')
        placeholders = soup.find_all('div', {'class': 'placeholder'})
        for x in placeholders:
            comment = ''.join(x.next_siblings)
            soup_comment = BS(comment, 'html.parser')
            tables = soup_comment.find_all('table', attrs={"id":"advanced"})
            for tag in tables:
                df = pd.read_html(tag.prettify())[0]
                if df.shape[0] > 3:
                    try:
                        table = df.iloc[2, :]
                        advance = advance.append(table).reset_index()
                        advance = advance[['Season', 'Age', 'Tm', 'Pos', 'G',
                                            'MP', 'TS%', '3PAr', 'FTr', 'ORB%',
                                            'DRB%', 'TRB%', 'AST%', 'STL%',
                                            'BLK%', 'TOV%', 'USG%', 'OWS', 'DWS',
                                            'WS', 'WS/48', 'OBPM', 'DBPM', 'BPM',
                                            'VORP']]
                    except:
                        print('No record for {0} {1}'.format(row['first_name'],
                              row['last_name']))
                else:
                    print('No record for {0} {1}'.format(row['first_name'],
                              row['last_name']))
    return advance


if __name__=="__main__":
    # Scrape Drafted Players Since 2011 and clean names
    draft_picks = draft_picks()
    draft_picks = player_names(draft_picks)

    # Save .csv of drafted player names
    draft_picks.to_csv('../data/draft_picks.csv')

    # Load Meta Data
    meta_data = pd.read_csv('../data/meta_data.csv')

    # Save .csv of college per100 data
    per_100 = college_per100(meta_data)
    per_100.to_csv('../data/college_per100.csv')

    # Save .csv of college advance data
    advance = college_advance(meta_data)
    advance.to_csv('../data/college_advance.csv')

    # Save .csv of nba advance data
    advance_nba = nba_advance(meta_data)
    advance_nba.to_csv('../data/nba_advance.csv')
