import numpy as np

def stage_cost(xx, xx_ref, uu, uu_ref, QQ, RR):

  xx = xx.reshape((-1, 1))
  xx_ref = xx_ref.reshape((-1, 1))
  uu = uu.reshape((-1, 1))
  uu_ref = uu_ref.reshape((-1, 1))

 
  JJ = 0.5*(xx - xx_ref).T@QQ@(xx - xx_ref) + 0.5*(uu - uu_ref).T@RR@(uu - uu_ref)
 

  dJ1 = QQ@(xx - xx_ref)
  dJ2 = RR@(uu - uu_ref)

  ddJ11 = QQ
  ddJ12 = np.zeros((3, 4))
  ddJ22 = RR

 
  return JJ.squeeze(), dJ1.squeeze(), dJ2.squeeze(), ddJ11.squeeze(), ddJ12.squeeze(), ddJ22.squeeze()

def term_cost(xx, xx_ref, QQT):

  xx = xx.reshape((-1, 1))
  xx_ref = xx_ref.reshape((-1, 1))


  JJT = 0.5*(xx - xx_ref).T@QQT@(xx - xx_ref)
  dJT1 = QQT@(xx - xx_ref)
  ddJT11 = QQT


  return JJT.squeeze(), dJT1.squeeze(), ddJT11.squeeze()
