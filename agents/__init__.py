from agents.acfql import ACFQLAgent
from agents.acrlpd import ACRLPDAgent
from agents.debs import DEBSAgent
from agents.resf import RESFAgent
from agents.cfgrl import CFGRLAgent
from agents.fql import FQLAgent
from agents.addf import ADDFAgent

agents = dict(
    acfql=ACFQLAgent,
    acrlpd=ACRLPDAgent,
    debs=DEBSAgent,
    resf=RESFAgent,
    cfgrl=CFGRLAgent,
    fql=FQLAgent,
    addf=ADDFAgent
)
