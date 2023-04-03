import pandas as pd
import numpy as np
import calendar
from typing import Union,Tuple
from sber_housing.constants import TARGET_COL


def run(df: pd.DataFrame, return_y: bool = False) -> Union[Tuple[pd.DataFrame, pd.Series], pd.DataFrame]:
    """
    creates new features for raw data input
    :param df: input raw data
    :param return_y: if True return features and target if the latter exists 
    :return: processed features ready for model training

    """
    df = df.copy()
    # converting timestamp to daretime object


    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['year']  = df['timestamp'].dt.year
    df['month'] = df['timestamp'].dt.month

    # adding new features month and year and day

    df['month_sin'] = np.sin(df['month'] / 12 * 2 * np.pi)
    df['month_cos'] = np.cos(df['month'] / 12 * 2 * np.pi)
    df['day'] = df['timestamp'].dt.day

    def num_days(dt):
        return calendar.monthrange(dt.year,dt.month)[1]

    days = df['timestamp'].apply(num_days)
    df['day_sin'] = np.sin(df['day'] / days * 2 * np.pi)
    df['day_cos'] = np.cos(df['day'] / days * 2 * np.pi)



    df['ecology'] = df['ecology'].map({'poor':1,'satisfactory':2,'good':3,'excellent':4})
    df['ecology'] = df['ecology'].fillna(df['ecology'].mean())

    df.loc[df['full_sq']>2000,'full_sq'] = np.nan
    df.loc[df['full_sq']<10,'full_sq'] = np.nan
    df.loc[df['life_sq']>500,'life_sq'] = np.nan
    df.loc[df['life_sq']<10,'life_sq'] = np.nan
    df.loc[df['kitch_sq']>175,'kitch_sq'] = np.nan
    df.loc[df['life_sq']>df['full_sq'],'life_sq'] = np.nan
    df.loc[df['kitch_sq']>df['full_sq'],'kitch_sq'] = np.nan
    df.loc[df['state'] > 4,'state'] = np.nan
    #df.loc[df['max_floor']==0,'max_floor'] = 1
    #df.loc[df['floor']==0,'floor'] = 1
    df.loc[df['max_floor']<df['floor'],'max_floor'] = np.nan
    df.loc[df['max_floor']==117,'max_floor'] = np.nan
    df.loc[df['build_year']<1800,'build_year'] = np.nan
    df.loc[df['build_year']>2030,'build_year'] = np.nan
    df.loc[df['preschool_quota']==0,'preschool_quota']= np.nan
    df['product_type']=df['product_type'].fillna('Investment')


    df['avg_room_sq'] = df['life_sq']/df['num_room']
    df['life_sq_ratio'] = df['life_sq']/df['full_sq']
    df['kitch_sq_ratio'] = df['kitch_sq']/df['full_sq']
    df['popul_density'] = df['raion_popul']/df['area_m']
    df['employment_ratio'] = df['work_all']/df['full_all']
    df['preschool_ratio_quota'] = df['preschool_quota']/df['children_preschool']
    df['school_ratio_quota'] = df['school_quota']/df['children_school']


    tsao_list = ["Basmannoe","Meshhanskoe","Arbat","Zamoskvorech'e","Hamovniki","Jakimanka","Krasnosel'skoe","Presnenskoe","Taganskoe","Tverskoe"]
    yzao_list = ["Akademicheskoe","Gagarinskoe","Lomonosovskoe","Cheremushki","Jasenevo","Kon'kovo","Kotlovka","Severnoe Butovo","Teplyj Stan","Zjuzino","Juzhnoe Butovo","Obruchevskoe"]
    yuvao_list = ["Kapotnja","Mar'ino","Nekrasovka","Tekstil'shhiki","Vyhino-Zhulebino","Kuz'minki","Nizhegorodskoe","Juzhnoportovoe","Lefortovo","Ljublino","Pechatniki","Rjazanskij"]
    yuao_list = ["Caricyno","Chertanovo Juzhnoe","Nagatino-Sadovniki","Zjablikovo","Birjulevo Vostochnoe","Brateevo","Chertanovo Severnoe","Donskoe","Orehovo-Borisovo Juzhnoe","Moskvorech'e-Saburovo","Nagatinskij Zaton","Birjulevo Zapadnoe","Chertanovo Central'noe","Danilovskoe","Nagornoe","Orehovo-Borisovo Severnoe"]
    tao_list = ["Poselenie Mihajlovo-Jarcevskoe","Poselenie Novofedorovskoe","Poselenie Rogovskoe","Troickij okrug","Poselenie Kievskij","Poselenie Krasnopahorskoe","Poselenie Pervomajskoe","Poselenie Shhapovskoe","Poselenie Klenovskoe","Poselenie Voronovskoe","Poselenie Pervomajskoe"]
    szao_list = ["Kurkino","Mitino","Pokrovskoe Streshnevo","Severnoe Tushino","Strogino","Shhukino","Horoshevo-Mnevniki","Juzhnoe Tushino"]
    svao_list = ["Altuf'evskoe","Bibirevo","Jaroslavskoe","Juzhnoe Medvedkovo","Ostankinskoe","Rostokino","Severnoe","Severnoe Medvedkovo","Lianozovo","Losinoostrovskoe","Marfino","Otradnoe","Sviblovo","Alekseevskoe","Babushkinskoe","Mar'ina Roshha","Butyrskoe"]
    sao_list = ["Ajeroport","Dmitrovskoe","Levoberezhnoe","Molzhaninovskoe","Begovoe","Golovinskoe","Horoshevskoe","Sokol","Vostochnoe Degunino","Beskudnikovskoe","Hovrino","Koptevo","Savelovskoe","Timirjazevskoe","Vojkovskoe","Zapadnoe Degunino"]
    nmao_list = ["Poselenie Filimonkovskoe","Poselenie Kokoshkino","Poselenie Sosenskoe","Poselenie Voskresenskoe","Poselenie Moskovskij","Poselenie Vnukovskoe","Poselenie Desjonovskoe","Poselenie Marushkinskoe","Poselenie Mosrentgen","Poselenie Rjazanovskoe","Poselenie Shherbinka"]
    zelao_list = ["Krjukovo","Silino","Savelki" ,"Matushkino","Staroe Krjukovo"]
    zao_list = ["Filevskij Park","Ochakovo-Matveevskoe","Prospekt Vernadskogo","Fili Davydkovo","Krylatskoe","Ramenki","Solncevo","Troparevo-Nikulino","Vnukovo","Dorogomilovo","Kuncevo","Mozhajskoe","Novo-Peredelkino"]
    vao_list = ["Bogorodskoe","Gol'janovo","Ivanovskoe","Kosino-Uhtomskoe","Novogireevo","Perovo","Sokolinaja Gora","Veshnjaki","Vostochnoe","Izmajlovo","Metrogorodok","Novokosino","Preobrazhenskoe","Severnoe Izmajlovo","Sokol'niki","Vostochnoe Izmajlovo"]

    df['AO'] = (
        df['sub_area']
        .where(~df['sub_area'].isin(tsao_list),'TsAO')
        .where(~df['sub_area'].isin(yzao_list),'YZAO')
        .where(~df['sub_area'].isin(yuvao_list),'YVAO')
        .where(~df['sub_area'].isin(yuao_list),'YUAO')
        .where(~df['sub_area'].isin(tao_list),'TAO')
        .where(~df['sub_area'].isin(szao_list),'SZAO')
        .where(~df['sub_area'].isin(svao_list),'SVAO')
        .where(~df['sub_area'].isin(sao_list),'SAO')
        .where(~df['sub_area'].isin(nmao_list),'NMAO')
        .where(~df['sub_area'].isin(zelao_list),'ZelAO')
        .where(~df['sub_area'].isin(zao_list),'ZAO')
        .where(~df['sub_area'].isin(vao_list),'VAO')
    )

    df['house_years'] = df['year']-df['build_year']
    df = df.drop(columns='build_year')

    df.loc[df['preschool_quota']==0,'preschool_quota']= np.nan

    df = df.drop(columns = 'timestamp')
    try:
        y=df[TARGET_COL]
    except KeyError:
        y=None
    X = df.drop(columns=TARGET_COL,errors='ignore')

    if return_y:
        return X, y
    else:
        return X
