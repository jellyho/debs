from agents.acfql import ACFQLAgent
from agents.fql import FQLAgent
from agents.hlmeanflowq import HLMEANFLOWQAgent
from agents.meanflowq import MEANFLOWQAgent
from agents.qcmfql import QCMFQLAgent
from agents.meanflow import MEANFLOWAgent
from agents.flow import FLOWAgent

agents = dict(
    acfql=ACFQLAgent,
    fql=FQLAgent,
    hlmeanflowq=HLMEANFLOWQAgent,
    meanflowq=MEANFLOWQAgent,
    qcmfql=QCMFQLAgent,
    flow=FLOWAgent,
    meanflow=MEANFLOWAgent
)
