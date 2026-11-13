# DARE — links

## Slides / poster
- Poster (final): dare-handout-final.pdf  (from ecrc/dare misc/)
  https://github.com/ecrc/dare/blob/master/misc/dare-handout-final.pdf
- Poster (draft PNG): dare-handout-draft2.png
  https://github.com/ecrc/dare/blob/master/misc/dare-handout-draft2.png

## Code
- GitHub: https://github.com/ecrc/dare
- Zenodo (install scripts only, v0.1, 2022-11-10): https://zenodo.org/records/7309032
  DOI: 10.5281/zenodo.7309032
- Hard real-time panel (TLR-MVM, separate repo): https://github.com/ecrc/tlrmvm

## Closed-loop simulation

- Jesse Cranney's COMPASS fork is `jcranney/shesha`. The current upstream COMPASS repository is on GitLab. [Fork](https://github.com/jcranney/shesha) · [Upstream](https://gitlab.obspm.fr/cosmic-rtc/compass)
- COMPASS's `generic_linear` controller applies the innovation gain with `set_L_matrix`. Cranney added the public example configuration in 2021. [Controller code](https://gitlab.obspm.fr/cosmic-rtc/compass/-/blob/d654d4788a246909eeea104312c21dad02283d5d/libsutra/src.cpp/sutra_controller_generic_linear.cpp#L242) · [Python API](https://gitlab.obspm.fr/cosmic-rtc/compass/-/blob/d654d4788a246909eeea104312c21dad02283d5d/shesha/shesha/supervisor/components/rtc/rtcAbstract.py#L463) · [Cranney's example commit](https://gitlab.obspm.fr/cosmic-rtc/compass/-/commit/9811bd58e8020f9340681c19a1c6d728d8f12f36)
- Cranney's MAVIS paper describes the Riccati solve and the per-frame estimator. The `descK` export in DARE is commented out. The public COMPASS example configures the controller with synthetic dimensions. The MAVIS setup and FITS loader remain unverified. [MAVIS paper](https://ao4elt6.copl.ulaval.ca/proceedings/401-qDC4-251.pdf) · [DARE export](dare/ddarechameleon.cpp#L674)

## Dataset (not public)
- KAUST SharePoint (Hatem Ltaief, KAUST login likely required):
  https://kaust-my.sharepoint.com/:f:/g/personal/ltaiefh_kaust_edu_sa/EgaqEKCdWURBgDq3x2CNsXQBteQs4x0VIdiiHJcY8AVyVA?e=EzETNF
- Expected files per size N (7090, 12570, 19610, 28210):
  At{N}_transpose.fits, BinvRt{N}_transpose.fits, Btinitial{N}_transpose.fits, Qt{N}.fits, Rt{N}.fits
- Original raw source: Feb21_DARE_matrices.mat + Feb21_gamma_hist.fits (used by misc/dare.m)
- Old KAUST path: /home/hongy0a/data/astronomydare/<N>/

## Related decks (Google Drive, AO context, no DARE slides)
- ltaief_sc21.pptx (TLR-MVM, SC'21): https://drive.google.com/file/d/1YPZbcjmzjad1oNJUpKKGW_oALbBviFZi/view
- 24jobtalk.pptx: https://drive.google.com/file/d/1RZ2qWhab1h3ZrshLoR0r28xiXSxVI5UD/view
