from agents.qcfql import QCFQLAgent
# from agents.hlmeanflowq import HLMEANFLOWQAgent
from agents.meanflowq import MEANFLOWQAgent
from agents.qcmfql import QCMFQLAgent
from agents.meanflow import MEANFLOWAgent
from agents.flow import FLOWAgent
from agents.fmlql import FMLQLAgent
from agents.cfm import ConsistencyFlowAgent
from agents.meanflowrf import MEANFLOWRFAgent
from agents.dsrl import DSRLAgent

agents = dict(
    qcfql=QCFQLAgent,
    fmlql=FMLQLAgent,
    meanflowq=MEANFLOWQAgent,
    qcmfql=QCMFQLAgent,
    flow=FLOWAgent,
    meanflow=MEANFLOWAgent,
    cfm=ConsistencyFlowAgent,
    meanflowrf=MEANFLOWRFAgent,
    dsrl=DSRLAgent,
)
