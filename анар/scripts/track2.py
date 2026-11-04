import cv2, json, numpy as np, os
from ai_edge_litert.interpreter import Interpreter
it=Interpreter('/tmp/pose_landmarks_detector.tflite',num_threads=8);it.allocate_tensors()
ii=it.get_input_details()[0]['index']; od={d['name']:d['index'] for d in it.get_output_details()}
def run(fr,cx,cy,s):
    M=np.float32([[256/s,0,128-cx*256/s],[0,256/s,128-cy*256/s]])
    crop=cv2.warpAffine(fr,M,(256,256),borderValue=0)
    it.set_tensor(ii,(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)[None]/255.).astype(np.float32)); it.invoke()
    L=it.get_tensor(od['Identity'])[0].reshape(39,5); seg=it.get_tensor(od['Identity_2'])[0,:,:,0]
    pts=np.c_[(L[:,0]-128)*s/256+cx,(L[:,1]-128)*s/256+cy,1/(1+np.exp(-L[:,3]))]
    segf=cv2.warpAffine((1/(1+np.exp(-seg))).astype(np.float32),M,(fr.shape[1],fr.shape[0]),flags=cv2.WARP_INVERSE_MAP)
    return pts,segf
import sys; FX,FY,FS=map(float,sys.argv[1:4]); cap=cv2.VideoCapture('seg.mp4'); P=[]; os.makedirs('masks2',exist_ok=True); i=0
while True:
    ok,fr=cap.read()
    if not ok: break
    pts,segf=run(fr,FX,FY,FS)
    P.append(pts.tolist()); cv2.imwrite(f'masks2/{i:04d}.png',(segf*255).astype(np.uint8)); i+=1
json.dump(P,open('pose2.json','w')); print(i)
