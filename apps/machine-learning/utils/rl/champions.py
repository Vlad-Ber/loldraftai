# TODO: should just restrict match prediction to consecutive ids, to avoid having to filter ids that go to 900.
VALID_CHAMPION_IDS = [
    1,  # Annie
    2,  # Olaf
    3,  # Galio
    4,  # Twisted Fate
    5,  # Xin Zhao
    6,  # Urgot
    7,  # LeBlanc
    8,  # Vladimir
    9,  # Fiddlesticks
    10,  # Kayle
    11,  # Master Yi
    12,  # Alistar
    13,  # Ryze
    14,  # Sion
    15,  # Sivir
    16,  # Soraka
    17,  # Teemo
    18,  # Tristana
    19,  # Warwick
    20,  # Nunu & Willump
    21,  # Miss Fortune
    22,  # Ashe
    23,  # Tryndamere
    24,  # Jax
    25,  # Morgana
    26,  # Zilean
    27,  # Singed
    28,  # Evelynn
    29,  # Twitch
    30,  # Karthus
    31,  # Cho'Gath
    32,  # Amumu
    33,  # Rammus
    34,  # Anivia
    35,  # Shaco
    36,  # Dr. Mundo
    37,  # Sona
    38,  # Kassadin
    39,  # Irelia
    40,  # Janna
    41,  # Gangplank
    42,  # Corki
    43,  # Karma
    44,  # Taric
    45,  # Veigar
    48,  # Trundle
    50,  # Swain
    51,  # Caitlyn
    53,  # Blitzcrank
    54,  # Malphite
    55,  # Katarina
]

# Champion role constants with names as comments
TOP_CHAMPIONS = [
    2,  # Olaf
    6,  # Urgot
    8,  # Vladimir
    13,  # Ryze
    10,  # Kayle
    14,  # Sion
    17,  # Teemo
    19,  # Warwick
    23,  # Tryndamere
    24,  # Jax
    27,  # Singed
    31,  # Cho'Gath
    36,  # Dr. Mundo
    39,  # Irelia
    41,  # Gangplank
    48,  # Trundle
    54,  # Malphite
]

JUNGLE_CHAMPIONS = [
    2,  # Olaf
    5,  # Xin Zhao
    9,  # Fiddlesticks
    11,  # Master Yi
    19,  # Warwick
    20,  # Nunu & Willump
    28,  # Evelynn
    35,  # Shaco
    48,  # Trundle
]

MID_CHAMPIONS = [
    1,  # Annie
    3,  # Galio
    4,  # Twisted Fate
    18,  # Tristana
    7,  # LeBlanc
    8,  # Vladimir
    13,  # Ryze
    31,  # Cho'Gath
    33,  # Rammus
    38,  # Kassadin
    39,  # Irelia
    42,  # Corki
    55,  # Katarina
]

BOT_CHAMPIONS = [
    15,  # Sivir
    18,  # Tristana
    21,  # Miss Fortune
    22,  # Ashe
    29,  # Twitch
    30,  # Karthus
    50,  # Swain
    51,  # Caitlyn
]

UTILITY_CHAMPIONS = [
    12,  # Alistar
    16,  # Soraka
    25,  # Morgana
    26,  # Zilean
    37,  # Sona
    40,  # Janna
    43,  # Karma
    44,  # Taric
    53,  # Blitzcrank
]

# Combine all role-based champions into a single list
ROLE_CHAMPIONS = [
    TOP_CHAMPIONS,
    JUNGLE_CHAMPIONS,
    MID_CHAMPIONS,
    BOT_CHAMPIONS,
    UTILITY_CHAMPIONS,
]
