import sys
sys.path.append("../")
import time as dt
from backend_scripts.utils import get_default_hp
from backend_scripts.tasks import npapa_script


if __name__ == '__main__':

    hp = get_default_hp(task='12pa',platform='server')
    hp['agenttype'] = 'sym'
    hp['btstp'] = 15
    hp['savefig'] = True
    hp['savevar'] = False
    hp['savegenvar'] = False

    hp['recallbeta'] = 1

    hp['usesmc'] = True

    hp['render'] = False  # visualise movement trial by trial

    hp['exptname'] = '{}_{}_xy_{}smc_{}t_{}om_{}taua_{}xy_b{}_{}'.format(hp['usesmc'],
        hp['task'], hp['agenttype'],hp['time'], hp['omitg'], hp['taua'],hp['xylr'],hp['btstp'],dt.monotonic())

    totdgr, totpi, mvpath = npapa_script(hp)