import numpy as np
import cv2
import scipy as sc
import pandas as pd
import PIL
import re

from os import listdir
from os.path import isfile, join
from skimage.io import imread, imsave
from pandas import DataFrame, Series
from numpy.random import RandomState 
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from datetime import datetime

startTime = datetime.now()

#00 - Adjustable parameters
filePath = '../../01_TIF/'
cascPath = '../02_classifier_training/classifier/cascade.xml'

frame_rate = 0.040 #seconds
pixel_dimension = 0.694 #micro meter

numberClosestFrames = 50 # the number of frames to average

detectParticlesScaleFactor = 1.2 #detection window scale factor
detectParticlesMinNeighbors = 140 #min number of neighbors for region to be positive
detectParticlesMinSize = (36, 68) #min size for search window
detectParticlesMaxSize = (63, 119) #max size for search window

numberOfLookaheadFrames = 2 #number of frames to lookahead for candidate particles
xRangeCandidateParticles = 30 #range of x values to search for candidate particles
minimumTrajectoryLength = 10 #minimum length for trajectory to be considered valid


imageList = sorted([ f for f in listdir(filePath) if isfile(join(filePath,f)) ])#[:300] #debugging
numberOfFrames = len(imageList)
experimentName  = re.sub('(_\d+\.\w+)', '', imageList[0])
fontPath = "/usr/share/fonts/truetype/ubuntu-font-family/Ubuntu-R.ttf"
fontSize = 20

clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(8,8))

assert numberOfFrames >= numberClosestFrames * 2

#01 - Define functions
closest_frames_start = np.array([ imread(filePath+imageList[f]) for f in range(0, 2*numberClosestFrames+1) ])
closest_frames_middle = np.array([ imread(filePath+imageList[f]) for f in range(0, 2*numberClosestFrames+1) ])
closest_frames_end = np.array([ imread(filePath+imageList[f]) for f in range(numberOfFrames-2*numberClosestFrames, numberOfFrames) ])
def subtract_background(active_frame, numberClosestFrames, imageList,  numberOfFrames, closest_frames_start, closest_frames_middle, closest_frames_end):
    "docstring"
    if frame <= numberClosestFrames: #start frames
        absolute = np.array(np.absolute(active_frame - closest_frames_start.mean(axis=0)), dtype=active_frame.dtype)
    if frame > numberOfFrames - numberClosestFrames - 3: #end frames
        absolute = np.array(np.absolute(active_frame - closest_frames_end.mean(axis=0)), dtype=active_frame.dtype)
    else: #middle frames
        concat_frame =  np.array(imread(filePath+imageList[frame+numberClosestFrames+2]))
        closest_frames_middle = np.concatenate([np.delete(closest_frames_middle, 0, axis=0),concat_frame.reshape((1,)+concat_frame.shape)], axis=0)
        absolute = np.array(np.absolute(active_frame - closest_frames_middle.mean(axis=0)), dtype=active_frame.dtype)
    absolute *= (255/absolute.max())
    absolute = clahe.apply(absolute)
    return (absolute, closest_frames_middle)

def detect_particles(frame, cascPath, detectParticlesScaleFactor, detectParticlesMinNeighbors, detectParticlesMinSize, detectParticlesMaxSize):
    "OpenCV haar features detectMultiScale classifier, window search applied to image returning particle coordinates and rectangle"
    particles_frame = cv2.CascadeClassifier(cascPath).detectMultiScale(frame,scaleFactor=detectParticlesScaleFactor,minNeighbors=detectParticlesMinNeighbors,minSize=detectParticlesMinSize,maxSize=detectParticlesMaxSize,flags = cv2.cv.CV_HAAR_SCALE_IMAGE)
    for (x, y, w, h) in particles_frame:
        cv2.rectangle(absolute, (x, y), (x+w, y+h), (255, 255, 255), 1)
        cv2.rectangle(active_frame, (x, y), (x+w, y+h), (0, 0, 0), 1)
    return (particles_frame, absolute, active_frame)

def particle_chooser(df_particles, index, numberOfLookaheadFrames):
    "Selects particle to link in particles data frame, uses w search applied to image returning particle coordinates and rectangle"
    particle = df_particles.loc[[index]]
    lookahead_frames = df_particles.loc[df_particles.frame_no.isin(range(int(particle.frame_no+1),int(particle.frame_no+numberOfLookaheadFrames)))]
    candidate_particles = lookahead_frames[(lookahead_frames['x_centroid'] > int(particle.x_centroid)-xRangeCandidateParticles) & (lookahead_frames['x_centroid'] < int(particle.x_centroid)+xRangeCandidateParticles)]
    candidate_particles = candidate_particles[(candidate_particles['y_centroid'] > int(particle.y_centroid))]
    candidate_particles = candidate_particles.sort(['y_centroid'], ascending=[True])
    chosen_particle = candidate_particles[:1]
    chosen_particle['particle_ID'] = int(particle['particle_ID'])
    if chosen_particle.shape[0]>0:
        df_particles.loc[chosen_particle.index[0],'particle_ID'] = int(chosen_particle['particle_ID'])
    return chosen_particle

#02 - Detect particles
particles = np.empty((0,5), int)
particles_columns = ['frame_no','x','y','w','h']

for frame in range(0, numberOfFrames):
    active_frame = np.asarray(imread(filePath+imageList[frame]))
    absolute, closest_frames_middle = subtract_background(active_frame, numberClosestFrames, imageList,  numberOfFrames, closest_frames_start, closest_frames_middle, closest_frames_end)
    cv2.imwrite('../04_python_results/01_absolute/'+imageList[frame].split('.',1)[0]+'.jpg', absolute)
    particles_frame, absolute, active_frame = detect_particles(absolute, cascPath, detectParticlesScaleFactor, detectParticlesMinNeighbors, detectParticlesMinSize, detectParticlesMaxSize)
    cv2.imwrite('../04_python_results/02_detection_absolute/'+imageList[frame].split('.',1)[0]+'.jpg', absolute)
    cv2.imwrite('../04_python_results/03_detection/'+imageList[frame].split('.',1)[0]+'.jpg', clahe.apply(active_frame))
    if len(particles_frame) > 0:
        frame_number = np.empty(particles_frame.shape[0])[...,None]; frame_number.fill(frame+1)
        particles_frame = np.hstack((frame_number, particles_frame))
        particles = np.append(particles, particles_frame)
        particles = particles.reshape(len(particles)/5, 5).astype(int)
        df_particles = pd.DataFrame.from_records(particles, columns=particles_columns)
        df_particles.to_csv('../04_python_results/'+experimentName+'_particle_features.csv', sep=',', index=False)
    print "Frame {0} of {1}, detected {2} particles".format(frame+1, numberOfFrames, len(particles_frame))
x_centroid = df_particles.x + (df_particles.w / 2); df_particles['x_centroid'] = x_centroid
y_centroid = df_particles.y + (df_particles.h / 2); df_particles['y_centroid'] = y_centroid
df_particles = df_particles.sort(['frame_no', 'y_centroid'], ascending=[True, True]).reset_index(drop=True)
df_particles["particle_ID"] = Series(range(1,df_particles.loc[df_particles['frame_no'] == df_particles['frame_no'].min()].shape[0]+1))
df_particles.drop('particle_ID', 1).to_csv('../04_python_results/'+experimentName+'_particle_features.csv', sep=',', index=False)

#03 - Track particles
for index, row in df_particles.iterrows():
    if pd.isnull(df_particles.loc[index, 'particle_ID']) == True:
        max_particle_ID = df_particles.particle_ID.max() 
        df_particles.loc[index, 'particle_ID'] = max_particle_ID + 1
        chosen_particle = particle_chooser(df_particles, index, numberOfLookaheadFrames)
    else:
        chosen_particle = particle_chooser(df_particles, index, numberOfLookaheadFrames)
    print "Linking particles {0} of {1}".format(index+1, len(df_particles))

#04 - Trajectories
uniqueParticles = np.unique(df_particles[['particle_ID']].values)
uniqueParticles = uniqueParticles.reshape(len(uniqueParticles),1)
seed = RandomState(9001)
particleColours = pd.DataFrame(np.concatenate((uniqueParticles, seed.randint(255, size=(len(uniqueParticles),3))), axis=1))
particleColours.columns = ['particle_ID', 'red', 'green', 'blue']
trajectories = pd.merge(df_particles, particleColours, on='particle_ID')
trajectories = trajectories[trajectories.groupby('particle_ID').particle_ID.transform(len) > minimumTrajectoryLength]
trajectories = trajectories[~trajectories['particle_ID'].isin(trajectories.loc[trajectories['frame_no'].isin([1,len(imageList)])]['particle_ID'].tolist())]
trajectories['particle_ID'] = trajectories['particle_ID'].rank('dense')
trajectories[['x_centroid_microns', 'y_centroid_microns']] = trajectories[['x_centroid', 'y_centroid']] * pixel_dimension
trajectories[['x_diff', 'y_diff']] = trajectories.groupby(['particle_ID'])[['x_centroid_microns', 'y_centroid_microns']].transform(lambda x: x.diff())
trajectories['velocity'] = np.sqrt(trajectories['x_diff']**2 + trajectories['y_diff']**2)/frame_rate
trajectories['acceleration'] = trajectories.groupby(['particle_ID'])['velocity'].transform(lambda x: x.diff())/frame_rate
trajectories.to_csv('../04_python_results/' + experimentName + '_trajectories.csv', sep=',', encoding='utf-8', index=False)
trajectories_summary = trajectories.groupby(['particle_ID'])
trajectories_summary = trajectories_summary[['frame_no', 'velocity', 'acceleration']].agg([len, np.median, np.mean, np.std, np.min, np.max]).T.drop_duplicates().T
trajectories_summary = trajectories_summary.drop(trajectories_summary.columns[[1,2]], axis=1)
trajectories_summary.columns = trajectories_summary.columns.droplevel(0)
trajectories_summary.columns = ['number_frames', 'start_frame', 'end_frame', 'velocity_median', 'velocity_mean', 'velocity_std', 'velocity_min', 'velocity_max', 'acceleration_median', 'acceleration_mean', 'acceleration_std', 'acceleration_min', 'acceleration_max']
middle_frame = ((trajectories_summary['start_frame'] + trajectories_summary['end_frame'])/2).round()
trajectories_summary.insert(2, 'middle_frame', ((trajectories_summary['start_frame'] + trajectories_summary['end_frame'])/2).round())
trajectories_summary.insert(0, 'particle_id', trajectories_summary.index)
trajectories_summary.to_csv('../04_python_results/' + experimentName + '_trajectories_summary.csv', sep=',', encoding='utf-8', index=False)

#05 - Output overlay frames
absolutePath = '../04_python_results/01_absolute/'
for frame in range(0, numberOfFrames):
    active_frame = Image.fromarray(clahe.apply(imread(filePath+imageList[frame]))).convert('RGBA')
    subtractedImages = Image.fromarray(imread(absolutePath+imageList[frame].split('.',1)[0]+'.jpg')).convert('RGBA')
    overlay = Image.new('RGBA', active_frame.size, (255,0,0,0))
    overlayDraw = ImageDraw.Draw(overlay)
    alpha = 100
    font = ImageFont.truetype(fontPath, fontSize)
    frame_particles = trajectories.loc[trajectories['frame_no'] == frame +1]
    number_particles_frame = int(frame_particles.particle_ID.count())
    for l in range(0, number_particles_frame):
        particle_ID = str(int(round(frame_particles.iloc[l].particle_ID)))
        x_centroid = int(round(frame_particles.iloc[l].x_centroid))
        y_centroid = int(round(frame_particles.iloc[l].y_centroid))
        w = int(round(frame_particles.iloc[l].w))
        h = int(round(frame_particles.iloc[l].h))
        r = int(round(frame_particles.iloc[l].red))
        g = int(round(frame_particles.iloc[l].green))
        b = int(round(frame_particles.iloc[l].blue))
        overlayDraw.rectangle((x_centroid - w/2, y_centroid - h/2, x_centroid + w/2, y_centroid + h/2), fill=(r, g, b, alpha/5))
        overlayDraw.line((x_centroid - w/2, y_centroid + h/2, x_centroid + w/2, y_centroid + h/2), fill=(r, g, b, alpha), width=2)
        overlayDraw.line((x_centroid - w/2, y_centroid - h/2, x_centroid + w/2, y_centroid - h/2), fill=(r, g, b, alpha), width=2)
        overlayDraw.line((x_centroid - w/2, y_centroid - h/2, x_centroid - w/2, y_centroid + h/2), fill=(r, g, b, alpha), width=2)
        overlayDraw.line((x_centroid + w/2, y_centroid - h/2, x_centroid + w/2, y_centroid + h/2), fill=(r, g, b, alpha), width=2)
        overlayDraw.text((x_centroid- w/2, y_centroid+h/2+10), particle_ID, (r, g, b), font)
    active_frame.paste(overlay, overlay)
    active_frame.save('../04_python_results/05_trajectory/'+imageList[frame].split('.',1)[0]+'.jpg')
    subtractedImages.paste(overlay, overlay)
    subtractedImages.save('../04_python_results/04_trajectory_absolute/'+imageList[frame].split('.',1)[0]+'.jpg')
    print "Saving trajectories frame {0} of {1}".format(frame+1, numberOfFrames)

print '\nAnalysis time: ', datetime.now() - startTime, ' seconds'
