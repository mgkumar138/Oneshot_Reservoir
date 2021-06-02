import sys
sys.path.append("../")
import time as dt
from backend_scripts.utils import get_default_hp
from backend_scripts.tasks import multiplepa_script


if __name__ == '__main__':

    hp = get_default_hp(task='6pa',platform='laptop')
    hp['agenttype'] = 'sym'
    hp['btstp'] = 1
    hp['savefig'] = True
    hp['savevar'] = False
    hp['savegenvar'] = False

    hp['recallbeta'] = 1

    hp['render'] = False  # visualise movement trial by trial

    hp['exptname'] = '{}_{}_xy_{}t_{}om_{}taua_{}xy_b{}_{}'.format(
        hp['task'], hp['agenttype'],hp['time'], hp['omitg'], hp['taua'],hp['xylr'],hp['btstp'],dt.monotonic())

    totlat, totdgr, totpi, mvpath = multiplepa_script(hp)
