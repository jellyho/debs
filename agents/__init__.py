from agents.acfql import ACFQLAgent
from agents.acrlpd import ACRLPDAgent
from agents.debs import DEBSAgent
from agents.resf import RESFAgent
from agents.cfgrl import CFGRLAgent
from agents.fql import FQLAgent
from agents.addf import ADDFAgent
from agents.hldebs import HLDEBSAgent
from agents.hlcfgrl import HLCFGRLAgent
from agents.hldsrl import HLDSRLAgent
from agents.hlmeanflow import HLMEANFLOWAgent
from agents.hlmeanflowq import HLMEANFLOWQAgent
from agents.hlmeanflowqg import HLMEANFLOWQGAgent

agents = dict(
    acfql=ACFQLAgent,
    acrlpd=ACRLPDAgent,
    debs=DEBSAgent,
    resf=RESFAgent,
    cfgrl=CFGRLAgent,
    fql=FQLAgent,
    addf=ADDFAgent,
    hldebs=HLDEBSAgent,
    hlcfgrl=HLCFGRLAgent,
    hlpriordsrl=HLDSRLAgent,
    hlmeanflow=HLMEANFLOWAgent,
    hlmeanflowq=HLMEANFLOWQAgent,
    hlmeanflowqg=HLMEANFLOWQGAgent,
)
