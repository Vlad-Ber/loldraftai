from enum import Enum


class Champion(Enum):
    ANNIE = (1, "Annie")
    OLAF = (2, "Olaf")
    GALIO = (3, "Galio")
    TWISTED_FATE = (4, "Twisted Fate")
    XIN_ZHAO = (5, "Xin Zhao")
    URGOT = (6, "Urgot")
    LEBLANC = (7, "LeBlanc")
    VLADIMIR = (8, "Vladimir")
    FIDDLESTICKS = (9, "Fiddlesticks")
    KAYLE = (10, "Kayle")
    MASTER_YI = (11, "Master Yi")
    ALISTAR = (12, "Alistar")
    RYZE = (13, "Ryze")
    SION = (14, "Sion")
    SIVIR = (15, "Sivir")
    SORAKA = (16, "Soraka")
    TEEMO = (17, "Teemo")
    TRISTANA = (18, "Tristana")
    WARWICK = (19, "Warwick")
    NUNU = (20, "Nunu & Willump")
    MISS_FORTUNE = (21, "Miss Fortune")
    ASHE = (22, "Ashe")
    TRYNDAMERE = (23, "Tryndamere")
    JAX = (24, "Jax")
    MORGANA = (25, "Morgana")
    ZILEAN = (26, "Zilean")
    SINGED = (27, "Singed")
    EVELYNN = (28, "Evelynn")
    TWITCH = (29, "Twitch")
    KARTHUS = (30, "Karthus")
    CHOGATH = (31, "Cho'Gath")
    AMUMU = (32, "Amumu")
    RAMMUS = (33, "Rammus")
    ANIVIA = (34, "Anivia")
    SHACO = (35, "Shaco")
    DR_MUNDO = (36, "Dr. Mundo")
    SONA = (37, "Sona")
    KASSADIN = (38, "Kassadin")
    IRELIA = (39, "Irelia")
    JANNA = (40, "Janna")
    GANGPLANK = (41, "Gangplank")
    CORKI = (42, "Corki")
    KARMA = (43, "Karma")
    TARIC = (44, "Taric")
    VEIGAR = (45, "Veigar")
    TRUNDLE = (48, "Trundle")
    SWAIN = (50, "Swain")
    CAITLYN = (51, "Caitlyn")
    BLITZCRANK = (53, "Blitzcrank")
    MALPHITE = (54, "Malphite")
    KATARINA = (55, "Katarina")

    def __init__(self, id, name):
        self.id = id
        self.name = name


# TODO: should just restrict match prediction to consecutive ids, to avoid having to filter ids that go to 900.
VALID_CHAMPION_IDS = [champion.id for champion in Champion]


# Champion role constants with names as comments
TOP_CHAMPIONS = [
    Champion.OLAF.id,
    Champion.URGOT.id,
    Champion.VLADIMIR.id,
    Champion.RYZE.id,
    Champion.KAYLE.id,
    Champion.SION.id,
    Champion.TEEMO.id,
    Champion.WARWICK.id,
    Champion.TRYNDAMERE.id,
    Champion.JAX.id,
    Champion.SINGED.id,
    Champion.CHOGATH.id,
    Champion.DR_MUNDO.id,
    Champion.IRELIA.id,
    Champion.GANGPLANK.id,
    Champion.TRUNDLE.id,
    Champion.MALPHITE.id,
]

JUNGLE_CHAMPIONS = [
    Champion.OLAF.id,
    Champion.XIN_ZHAO.id,
    Champion.FIDDLESTICKS.id,
    Champion.MASTER_YI.id,
    Champion.WARWICK.id,
    Champion.NUNU.id,
    Champion.EVELYNN.id,
    Champion.SHACO.id,
    Champion.TRUNDLE.id,
]

MID_CHAMPIONS = [
    Champion.ANNIE.id,
    Champion.GALIO.id,
    Champion.TWISTED_FATE.id,
    Champion.TRISTANA.id,
    Champion.LEBLANC.id,
    Champion.VLADIMIR.id,
    Champion.RYZE.id,
    Champion.CHOGATH.id,
    Champion.RAMMUS.id,
    Champion.KASSADIN.id,
    Champion.IRELIA.id,
    Champion.CORKI.id,
    Champion.KATARINA.id,
]

BOT_CHAMPIONS = [
    Champion.SIVIR.id,
    Champion.TRISTANA.id,
    Champion.MISS_FORTUNE.id,
    Champion.ASHE.id,
    Champion.TWITCH.id,
    Champion.KARTHUS.id,
    Champion.SWAIN.id,
    Champion.CAITLYN.id,
]

UTILITY_CHAMPIONS = [
    Champion.ALISTAR.id,
    Champion.SORAKA.id,
    Champion.MORGANA.id,
    Champion.ZILEAN.id,
    Champion.SONA.id,
    Champion.JANNA.id,
    Champion.KARMA.id,
    Champion.TARIC.id,
    Champion.BLITZCRANK.id,
]

# Combine all role-based champions into a single list
ROLE_CHAMPIONS = [
    TOP_CHAMPIONS,
    JUNGLE_CHAMPIONS,
    MID_CHAMPIONS,
    BOT_CHAMPIONS,
    UTILITY_CHAMPIONS,
]
