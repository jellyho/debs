from agents.qcfql import QCFQLAgent
from agents.fql import FQLAgent
from agents.hlmeanflowq import HLMEANFLOWQAgent
from agents.meanflowq import MEANFLOWQAgent
from agents.qcmfql import QCMFQLAgent
from agents.meanflow import MEANFLOWAgent
from agents.flow import FLOWAgent
from agents.fmlql import FMLQLAgent
from agents.meanflow_robot import MEANFLOWROBOTAgent

agents = dict(
    qcfql=QCFQLAgent,
    fmlql=FMLQLAgent,
    meanflowq=MEANFLOWQAgent,
    qcmfql=QCMFQLAgent,
    flow=FLOWAgent,
    meanflow=MEANFLOWAgent,
    meanflow_robot=MEANFLOWROBOTAgent
)
