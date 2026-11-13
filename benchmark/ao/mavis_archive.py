import shesha.config as conf
import numpy as np
simul_name = "mavis large"


###################
# LOOP
p_loop = conf.Param_loop()
p_loop.set_niter(10000)
p_loop.set_ittime(0.001)
###################
# GEOM
p_geom = conf.Param_geom()
p_geom.set_zenithangle(30.)
PUPDIAM = 360
p_geom.set_pupdiam(PUPDIAM)

###################
# TEL
p_tel = conf.Param_tel()
p_tel.set_diam(8.0)
p_tel.set_cobs(0.16)
p_tel.set_type_ap("VLT")
p_tel.set_t_spiders(-1)

###################
# ATMOS
p_atmos = conf.Param_atmos()
p_atmos.set_r0(0.125812) # equivalent of 0.8" at zenith
p_atmos.set_nscreens(10)
p_atmos.set_frac([0.59, 0.02, 0.04, 0.06, 0.01, 0.05, 0.09, 0.04,  0.05,  0.05])
p_atmos.set_alt([30,  140,  281,  562, 1125, 2250, 4500, 7750, 11000, 14000])
p_atmos.set_windspeed([ 6.6,  5.9,  5.1,  4.5,  5.1,  8.3, 16.3, 30.2,  34.3,  17.5])
p_atmos.set_winddir([0.,  10.,  20.,  25.,   0.,  10.,  20.,  25.,    0.,   10.])
p_atmos.set_L0([25.]*10)

###################
# TARGET
p_targets=[]
RADIUS_TAR = 15
if 1==1:
    NTAR = 10
    for t in np.arange(NTAR):
        p_targets.append(conf.Param_target())
        tar_rad = RADIUS_TAR*(np.sqrt(t/NTAR))
        #tar_rad = RADIUS_TAR*(t/NTAR)
        tar_x = tar_rad*np.cos(2.399963229*t)
        tar_y = tar_rad*np.sin(2.399963229*t)
        p_targets[t].set_xpos(tar_x)
        p_targets[t].set_ypos(tar_y)
        p_targets[t].set_Lambda( 0.55)
        p_targets[t].set_mag(10.)
        p_targets[t].set_dms_seen([0,1,2])
else:
    NTAR_X = 11
    NTAR = NTAR_X**2
    tar_x_array,tar_y_array = np.meshgrid(
            np.linspace(-RADIUS_TAR,RADIUS_TAR,NTAR_X),
            np.linspace(-RADIUS_TAR,RADIUS_TAR,NTAR_X))
    for t in np.arange(NTAR):
        p_targets.append(conf.Param_target())
        p_targets[t].set_xpos(tar_x_array.flatten()[t])
        p_targets[t].set_ypos(tar_y_array.flatten()[t])
        p_targets[t].set_Lambda( 0.55)
        p_targets[t].set_mag(10.)
        p_targets[t].set_dms_seen([0,1,2])
    # p_targets = p_targets[84:]


###################
# WFS
LGS_RADIUS  = 17.5
NGS_RADIUS  = 20.0
FRACSUB     = 0.8
NXSUB_LGS   = 40
NXSUB_NGS   = 1
NXSUB_TS    = 40
NXSUB_TAR   = max(NXSUB_LGS,NXSUB_NGS)
NLGS        = 8
NNGS        = 3
NTS         = 4

p_wfs_lgs = []
p_wfs_ngs = []
p_wfs_ts  = []

#lgs position
x = np.pi/8+np.linspace(0, 2 * np.pi, NLGS+1)[:-1]
lgs_xpos = LGS_RADIUS * np.cos(x)
lgs_ypos = LGS_RADIUS * np.sin(x)
LLT_RADIUS = 5.49 # metres
lltang = np.array([76.8,76.8,103.1,103.1,256.8,256.8,283.1,283.1])*np.pi/180
llt_xpos = LLT_RADIUS * np.cos(lltang)
llt_ypos = LLT_RADIUS * np.sin(lltang)

#NGS asterism
ngs_offset = 0.0
x = np.linspace(0,2 * np.pi, NNGS+1)[:-1]+ngs_offset
asterism_x = NGS_RADIUS * np.cos(x)
asterism_y = NGS_RADIUS * np.sin(x)

#TS position
NTS_X = 4
NTS = NTS_X**2
RADIUS_TS = 15.0
ts_x_pos,ts_y_pos = np.meshgrid(
        np.linspace(-RADIUS_TS,RADIUS_TS,NTS_X),
        np.linspace(-RADIUS_TS,RADIUS_TS,NTS_X))
ts_x_pos = ts_x_pos.flatten()
ts_y_pos = ts_y_pos.flatten()

#create wfs lists
#LGS
for i in range(NLGS):
    p_wfs_lgs.append(conf.Param_wfs())
#NGS
for i in range(NNGS):
    p_wfs_ngs.append(conf.Param_wfs())
#TS
for t in np.arange(NTS):
    p_wfs_ts.append(conf.Param_wfs())

#concatenate LGS and NGS
p_wfss = p_wfs_lgs + p_wfs_ngs

#setting LGS
for p_wfs in p_wfs_lgs:
    #TODO:
    # - add excess noise of 2 in variance
    # - add std=737nm of jitter to both tip and tilt before centroiding
    k=p_wfs_lgs.index(p_wfs)
    p_wfs.set_type("sh")
    p_wfs.set_nxsub(NXSUB_LGS)
    p_wfs.set_npix(6) 
    p_wfs.set_pixsize(0.833) 
    p_wfs.set_fracsub(FRACSUB)
    p_wfs.set_xpos(lgs_xpos[k])
    p_wfs.set_ypos(lgs_ypos[k])
    p_wfs.set_thetaML(11.25*k*np.pi/180)
    
    p_wfs.set_Lambda(0.589)
    p_wfs.set_gsmag(8.)
    p_wfs.set_optthroughput(0.5)
    p_wfs.set_zerop(1.e11)
    p_wfs.set_noise(0.2) #CCD220
    p_wfs.set_atmos_seen(1)

    p_wfs.set_gsalt(90 * 1.e3)
    p_wfs.set_lltx(llt_xpos[k])
    p_wfs.set_llty(llt_ypos[k])
    p_wfs.set_laserpower(17.05) 
    p_wfs.set_lgsreturnperwatt(22.)
    #p_wfs.set_proftype("Multipeak")
    p_wfs.set_beamsize(1.03) # higher than yao since we dont
                             # simulate uplink diffraction


#setting NGS
for p_wfs in p_wfs_ngs:
    #TODO:
    # - add excess noise of 1 in variance?
    # - dark current 20e-/s/pixel
    # - sky background 2144.4e-/m^2/arcsec^2/s?
    # - check flux == 1000ph/ms/sa?
    # - change centroider to wcog with 2pixel FWHM
    k=p_wfs_ngs.index(p_wfs)
    p_wfs.set_type("sh")
    p_wfs.set_nxsub(NXSUB_NGS)
    p_wfs.set_npix(20) 
    p_wfs.set_pixsize(0.03)
    p_wfs.set_fracsub(0.0)
    p_wfs.set_xpos(asterism_x[k])
    p_wfs.set_ypos(asterism_y[k])
    p_wfs.set_is_low_order(True)

    p_wfs.set_Lambda(1.65)
    p_wfs.set_gsmag(16.3-2)
    p_wfs.set_optthroughput(0.5) # /2.0 to emulate excess noise (variance of 2.0)
    p_wfs.set_zerop(1.e11)
    p_wfs.set_noise(0.5) #Saphira
    p_wfs.set_atmos_seen(1)

#setting TS
for p_ts in p_wfs_ts:
    k = p_wfs_ts.index(p_ts)
    p_ts.set_type("sh")
    p_ts.set_nxsub(NXSUB_TS)
    p_ts.set_npix(20) 
    p_ts.set_pixsize(0.833)
    p_ts.set_fracsub(0.8)
    p_ts.set_xpos(ts_x_pos[k])
    p_ts.set_ypos(ts_y_pos[k])
    p_ts.set_Lambda(0.589)
    p_ts.set_gsmag(16.3-2)
    p_ts.set_optthroughput(0.5) # /2.0 to emulate excess noise (variance of 2.0)
    p_ts.set_zerop(1.e11)
    p_ts.set_noise(0.5) #Saphira
    p_ts.set_atmos_seen(1)

###################
# DM
p_dm0 = conf.Param_dm()
p_dm1 = conf.Param_dm()
p_dm2 = conf.Param_dm()
p_dms = [p_dm0,p_dm1,p_dm2]

#p_dm0.set_type("pzt")
#p_dm0.set_file_influ_fits("DSM.fits")
#p_dm0.set_alt(0.)
#p_dm0.set_thresh(-1)
#p_dm0.set_unitpervolt(1)
#p_dm0.set_push4imat(0.1)
#p_dm0.set_keep_all_actu(True)

p_dm0.set_type("pzt")
p_dm0.set_nact(37)  # 37x37 covers the 8m pupil at 22cm pitch
p_dm0.set_alt(0.)
p_dm0.set_thresh(-1)
p_dm0.set_coupling(0.3)
p_dm0.set_unitpervolt(1)
p_dm0.set_push4imat(0.1)
p_dm0.set_keep_all_actu(True)
p_dm0._pitch = 0.22*PUPDIAM/8.0
# replace me DM0 picth

p_dm1.set_type("pzt")
#p_dm1.set_nact(46)  # 46x46 covers 120"x120" @ 6km with 25cm pitch
p_dm1.set_nact(39)   # 39x39 covers  60"x 60" @ 6km with 25cm pitch
p_dm1.set_alt(6000.)
p_dm1.set_thresh(-1)
p_dm1.set_coupling(0.3)
p_dm1.set_unitpervolt(1)
p_dm1.set_push4imat(0.1)
p_dm1.set_keep_all_actu(True)
p_dm1._pitch = 0.25*PUPDIAM/8.0
# replace me DM1 picth

p_dm2.set_type("pzt")
#p_dm2.set_nact(50)  # 50x50 covers 120"x120" @ 13.5km with 32cm pitch
p_dm2.set_nact(38)   # 38x38 covers  60"x 60" @ 13.5km with 32cm pitch
p_dm2.set_alt(13500.)
p_dm2.set_thresh(-1)
p_dm2.set_coupling(0.3)
p_dm2.set_unitpervolt(1)
p_dm2.set_push4imat(0.1)
p_dm2.set_keep_all_actu(True)
p_dm2._pitch = 0.32*PUPDIAM/8.0
# replace me DM2 picth

###################
# CENTROIDERS
p_centroiders = []
for i in range( NNGS+NLGS):
    p_centroiders.append(conf.Param_centroider())

for p_centroider in p_centroiders:
    k=p_centroiders.index(p_centroider)
    p_centroider.set_nwfs(k)
    if (p_wfss[k].get_gsalt() > 0):
        # LGS
        #p_centroider.set_type("cog")
        p_centroider.set_type("tcog")
        p_centroider.set_thresh(0.0)
        p_centroider.set_filter_TT(True)
    else:
        #NGS
        # TODO: Weighted cog
        p_centroider.set_type("cog")


###################
# CONTROLLERS
p_controller0 = conf.Param_controller()
p_controllers = [p_controller0]

p_controller0.set_type("generic_linear")
p_controller0.set_nwfs(np.arange(NLGS+NNGS))
p_controller0.set_ndm([0,1,2])
p_controller0.set_maxcond(150.)
p_controller0.set_delay(1.0)
p_controller0.set_gain(0.4)
