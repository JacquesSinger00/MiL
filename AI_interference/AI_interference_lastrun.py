#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on Tue 27 Jan 10:27:17 2026
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'untitled'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1920, 1200]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/jacquessinger/Documents/Academics/Research/BCI Projects/MIL_PsychoPy_fMRI/AI_interference/AI_interference_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=True,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    if deviceManager.getDevice('key_resp_2') is None:
        # initialise key_resp_2
        key_resp_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_2',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "trial" ---
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    instructions = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0, 0), draggable=False,      letterHeight=0.05,
         size=(1, 1), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='instructions',
         depth=-1, autoLog=True,
    )
    
    # --- Initialize components for Routine "sync_fMRI" ---
    key_resp_2 = keyboard.Keyboard(deviceName='key_resp_2')
    
    # --- Initialize components for Routine "rest" ---
    rest_fixation = visual.TextStim(win=win, name='rest_fixation',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "reading" ---
    textbox_coherence = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0, 0), draggable=False,      letterHeight=0.05,
         size=(0.9, 0.9), borderWidth=6.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=None, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=[1.0000, 1.0000, 1.0000],
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='textbox_coherence',
         depth=0, autoLog=True,
    )
    
    # --- Initialize components for Routine "task" ---
    textbox_task = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0, 0), draggable=False,      letterHeight=0.05,
         size=(0.9, 0.9), borderWidth=6.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=[-1.0000, 1.0000, -1.0000],
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='textbox_task',
         depth=0, autoLog=True,
    )
    
    # --- Initialize components for Routine "task_verification" ---
    task_verification_message = visual.TextStim(win=win, name='task_verification_message',
        text='Were you able to complete the task?\n\nPress 1 for yes and 2 for no. ',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "control" ---
    textbox = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0, 0), draggable=False,      letterHeight=0.05,
         size=(0.9, 0.9), borderWidth=6.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=[1.0000, -1.0000, -1.0000],
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='textbox',
         depth=0, autoLog=True,
    )
    
    # --- Initialize components for Routine "control_verification" ---
    text = visual.TextStim(win=win, name='text',
        text='Were there more than 16 vowels total in the previous sentences?\n\nPress 1 for yes and 2 for no. ',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "rest" ---
    rest_fixation = visual.TextStim(win=win, name='rest_fixation',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "reading2" ---
    reading2_textbox = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0, 0), draggable=False,      letterHeight=0.05,
         size=(0.9, 0.9), borderWidth=6.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=[1.0000, 1.0000, 1.0000],
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='reading2_textbox',
         depth=0, autoLog=True,
    )
    
    # --- Initialize components for Routine "task2" ---
    task2_textbox = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0, 0), draggable=False,      letterHeight=0.05,
         size=(0.9, 0.9), borderWidth=6.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=[-1.0000, 1.0000, -1.0000],
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='task2_textbox',
         depth=0, autoLog=True,
    )
    
    # --- Initialize components for Routine "task_verification" ---
    task_verification_message = visual.TextStim(win=win, name='task_verification_message',
        text='Were you able to complete the task?\n\nPress 1 for yes and 2 for no. ',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "control2" ---
    control2_textbox = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0, 0), draggable=False,      letterHeight=0.05,
         size=(0.9, 0.9), borderWidth=6.0,
         color=[1.0000, 1.0000, 1.0000], colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=[1.0000, -1.0000, -1.0000],
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='control2_textbox',
         depth=0, autoLog=True,
    )
    
    # --- Initialize components for Routine "control_verification" ---
    text = visual.TextStim(win=win, name='text',
        text='Were there more than 16 vowels total in the previous sentences?\n\nPress 1 for yes and 2 for no. ',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "rest" ---
    rest_fixation = visual.TextStim(win=win, name='rest_fixation',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "reading3" ---
    reading3_textbox = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0, 0), draggable=False,      letterHeight=0.05,
         size=(0.9, 0.9), borderWidth=6.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=[1.0000, 1.0000, 1.0000],
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='reading3_textbox',
         depth=0, autoLog=True,
    )
    
    # --- Initialize components for Routine "task3" ---
    task3_textbox = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0, 0), draggable=False,      letterHeight=0.05,
         size=(0.9, 0.9), borderWidth=6.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=[-1.0000, 1.0000, -1.0000],
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='task3_textbox',
         depth=0, autoLog=True,
    )
    
    # --- Initialize components for Routine "task_verification" ---
    task_verification_message = visual.TextStim(win=win, name='task_verification_message',
        text='Were you able to complete the task?\n\nPress 1 for yes and 2 for no. ',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "control3" ---
    control3_textbox = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Arial',
         ori=0.0, pos=(0, 0), draggable=False,      letterHeight=0.05,
         size=(0.9, 0.9), borderWidth=6.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=[1.0000, -1.0000, -1.0000],
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=False,
         name='control3_textbox',
         depth=0, autoLog=True,
    )
    
    # --- Initialize components for Routine "control_verification" ---
    text = visual.TextStim(win=win, name='text',
        text='Were there more than 16 vowels total in the previous sentences?\n\nPress 1 for yes and 2 for no. ',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "rest" ---
    rest_fixation = visual.TextStim(win=win, name='rest_fixation',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "end" ---
    end_message = visual.TextStim(win=win, name='end_message',
        text='You have completed the task.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # set up handler to look after randomisation of conditions etc
    in_instructions = data.TrialHandler2(
        name='in_instructions',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('../instructions.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(in_instructions)  # add the loop to the experiment
    thisIn_instruction = in_instructions.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisIn_instruction.rgb)
    if thisIn_instruction != None:
        for paramName in thisIn_instruction:
            globals()[paramName] = thisIn_instruction[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisIn_instruction in in_instructions:
        currentLoop = in_instructions
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisIn_instruction.rgb)
        if thisIn_instruction != None:
            for paramName in thisIn_instruction:
                globals()[paramName] = thisIn_instruction[paramName]
        
        # --- Prepare to start Routine "trial" ---
        # create an object to store info about Routine trial
        trial = data.Routine(
            name='trial',
            components=[key_resp, instructions],
        )
        trial.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for key_resp
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        instructions.reset()
        # store start times for trial
        trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trial.tStart = globalClock.getTime(format='float')
        trial.status = STARTED
        thisExp.addData('trial.started', trial.tStart)
        trial.maxDuration = None
        # keep track of which components have finished
        trialComponents = trial.components
        for thisComponent in trial.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial" ---
        # if trial has changed, end Routine now
        if isinstance(in_instructions, data.TrialHandler2) and thisIn_instruction.thisN != in_instructions.thisTrial.thisN:
            continueRoutine = False
        trial.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *key_resp* updates
            waitOnFlip = False
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.started')
                # update status
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                    key_resp.rt = _key_resp_allKeys[-1].rt
                    key_resp.duration = _key_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # *instructions* updates
            
            # if instructions is starting this frame...
            if instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                instructions.frameNStart = frameN  # exact frame index
                instructions.tStart = t  # local t and not account for scr refresh
                instructions.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(instructions, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'instructions.started')
                # update status
                instructions.status = STARTED
                instructions.setAutoDraw(True)
            
            # if instructions is active this frame...
            if instructions.status == STARTED:
                # update params
                instructions.setText(AI_instructions, log=False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                trial.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trial.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial" ---
        for thisComponent in trial.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trial
        trial.tStop = globalClock.getTime(format='float')
        trial.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trial.stopped', trial.tStop)
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
        in_instructions.addData('key_resp.keys',key_resp.keys)
        if key_resp.keys != None:  # we had a response
            in_instructions.addData('key_resp.rt', key_resp.rt)
            in_instructions.addData('key_resp.duration', key_resp.duration)
        in_instructions.addData('instructions.text',instructions.text)
        # the Routine "trial" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'in_instructions'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "sync_fMRI" ---
    # create an object to store info about Routine sync_fMRI
    sync_fMRI = data.Routine(
        name='sync_fMRI',
        components=[key_resp_2],
    )
    sync_fMRI.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_2
    key_resp_2.keys = []
    key_resp_2.rt = []
    _key_resp_2_allKeys = []
    # store start times for sync_fMRI
    sync_fMRI.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    sync_fMRI.tStart = globalClock.getTime(format='float')
    sync_fMRI.status = STARTED
    thisExp.addData('sync_fMRI.started', sync_fMRI.tStart)
    sync_fMRI.maxDuration = None
    # keep track of which components have finished
    sync_fMRIComponents = sync_fMRI.components
    for thisComponent in sync_fMRI.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "sync_fMRI" ---
    sync_fMRI.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *key_resp_2* updates
        waitOnFlip = False
        
        # if key_resp_2 is starting this frame...
        if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_2.frameNStart = frameN  # exact frame index
            key_resp_2.tStart = t  # local t and not account for scr refresh
            key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_2.started')
            # update status
            key_resp_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_2.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_2.getKeys(keyList=["5"], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_2_allKeys.extend(theseKeys)
            if len(_key_resp_2_allKeys):
                key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            sync_fMRI.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in sync_fMRI.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "sync_fMRI" ---
    for thisComponent in sync_fMRI.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for sync_fMRI
    sync_fMRI.tStop = globalClock.getTime(format='float')
    sync_fMRI.tStopRefresh = tThisFlipGlobal
    thisExp.addData('sync_fMRI.stopped', sync_fMRI.tStop)
    # check responses
    if key_resp_2.keys in ['', [], None]:  # No response was made
        key_resp_2.keys = None
    thisExp.addData('key_resp_2.keys',key_resp_2.keys)
    if key_resp_2.keys != None:  # we had a response
        thisExp.addData('key_resp_2.rt', key_resp_2.rt)
        thisExp.addData('key_resp_2.duration', key_resp_2.duration)
    thisExp.nextEntry()
    # the Routine "sync_fMRI" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler2(
        name='trials',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('../prompts.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            globals()[paramName] = thisTrial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial in trials:
        currentLoop = trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        
        # --- Prepare to start Routine "rest" ---
        # create an object to store info about Routine rest
        rest = data.Routine(
            name='rest',
            components=[rest_fixation],
        )
        rest.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for rest
        rest.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        rest.tStart = globalClock.getTime(format='float')
        rest.status = STARTED
        thisExp.addData('rest.started', rest.tStart)
        rest.maxDuration = None
        # keep track of which components have finished
        restComponents = rest.components
        for thisComponent in rest.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "rest" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        rest.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 12.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *rest_fixation* updates
            
            # if rest_fixation is starting this frame...
            if rest_fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                rest_fixation.frameNStart = frameN  # exact frame index
                rest_fixation.tStart = t  # local t and not account for scr refresh
                rest_fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rest_fixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rest_fixation.started')
                # update status
                rest_fixation.status = STARTED
                rest_fixation.setAutoDraw(True)
            
            # if rest_fixation is active this frame...
            if rest_fixation.status == STARTED:
                # update params
                pass
            
            # if rest_fixation is stopping this frame...
            if rest_fixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > rest_fixation.tStartRefresh + 12-frameTolerance:
                    # keep track of stop time/frame for later
                    rest_fixation.tStop = t  # not accounting for scr refresh
                    rest_fixation.tStopRefresh = tThisFlipGlobal  # on global time
                    rest_fixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rest_fixation.stopped')
                    # update status
                    rest_fixation.status = FINISHED
                    rest_fixation.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                rest.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in rest.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "rest" ---
        for thisComponent in rest.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for rest
        rest.tStop = globalClock.getTime(format='float')
        rest.tStopRefresh = tThisFlipGlobal
        thisExp.addData('rest.stopped', rest.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if rest.maxDurationReached:
            routineTimer.addTime(-rest.maxDuration)
        elif rest.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-12.000000)
        
        # --- Prepare to start Routine "reading" ---
        # create an object to store info about Routine reading
        reading = data.Routine(
            name='reading',
            components=[textbox_coherence],
        )
        reading.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        textbox_coherence.reset()
        # store start times for reading
        reading.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        reading.tStart = globalClock.getTime(format='float')
        reading.status = STARTED
        thisExp.addData('reading.started', reading.tStart)
        reading.maxDuration = None
        # keep track of which components have finished
        readingComponents = reading.components
        for thisComponent in reading.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "reading" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        reading.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 10.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *textbox_coherence* updates
            
            # if textbox_coherence is starting this frame...
            if textbox_coherence.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textbox_coherence.frameNStart = frameN  # exact frame index
                textbox_coherence.tStart = t  # local t and not account for scr refresh
                textbox_coherence.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textbox_coherence, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textbox_coherence.started')
                # update status
                textbox_coherence.status = STARTED
                textbox_coherence.setAutoDraw(True)
            
            # if textbox_coherence is active this frame...
            if textbox_coherence.status == STARTED:
                # update params
                textbox_coherence.setText(AI_text_01
                , log=False)
            
            # if textbox_coherence is stopping this frame...
            if textbox_coherence.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > textbox_coherence.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    textbox_coherence.tStop = t  # not accounting for scr refresh
                    textbox_coherence.tStopRefresh = tThisFlipGlobal  # on global time
                    textbox_coherence.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textbox_coherence.stopped')
                    # update status
                    textbox_coherence.status = FINISHED
                    textbox_coherence.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                reading.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in reading.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "reading" ---
        for thisComponent in reading.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for reading
        reading.tStop = globalClock.getTime(format='float')
        reading.tStopRefresh = tThisFlipGlobal
        thisExp.addData('reading.stopped', reading.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if reading.maxDurationReached:
            routineTimer.addTime(-reading.maxDuration)
        elif reading.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-10.000000)
        
        # --- Prepare to start Routine "task" ---
        # create an object to store info about Routine task
        task = data.Routine(
            name='task',
            components=[textbox_task],
        )
        task.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        textbox_task.reset()
        # store start times for task
        task.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        task.tStart = globalClock.getTime(format='float')
        task.status = STARTED
        thisExp.addData('task.started', task.tStart)
        task.maxDuration = None
        # keep track of which components have finished
        taskComponents = task.components
        for thisComponent in task.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "task" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        task.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 20.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *textbox_task* updates
            
            # if textbox_task is starting this frame...
            if textbox_task.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textbox_task.frameNStart = frameN  # exact frame index
                textbox_task.tStart = t  # local t and not account for scr refresh
                textbox_task.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textbox_task, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textbox_task.started')
                # update status
                textbox_task.status = STARTED
                textbox_task.setAutoDraw(True)
            
            # if textbox_task is active this frame...
            if textbox_task.status == STARTED:
                # update params
                textbox_task.setText(AI_text_01, log=False)
            
            # if textbox_task is stopping this frame...
            if textbox_task.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > textbox_task.tStartRefresh + 20-frameTolerance:
                    # keep track of stop time/frame for later
                    textbox_task.tStop = t  # not accounting for scr refresh
                    textbox_task.tStopRefresh = tThisFlipGlobal  # on global time
                    textbox_task.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textbox_task.stopped')
                    # update status
                    textbox_task.status = FINISHED
                    textbox_task.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                task.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in task.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "task" ---
        for thisComponent in task.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for task
        task.tStop = globalClock.getTime(format='float')
        task.tStopRefresh = tThisFlipGlobal
        thisExp.addData('task.stopped', task.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if task.maxDurationReached:
            routineTimer.addTime(-task.maxDuration)
        elif task.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-20.000000)
        
        # --- Prepare to start Routine "task_verification" ---
        # create an object to store info about Routine task_verification
        task_verification = data.Routine(
            name='task_verification',
            components=[task_verification_message],
        )
        task_verification.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code
        color_text = 'white'  # default
        # store start times for task_verification
        task_verification.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        task_verification.tStart = globalClock.getTime(format='float')
        task_verification.status = STARTED
        thisExp.addData('task_verification.started', task_verification.tStart)
        task_verification.maxDuration = None
        # keep track of which components have finished
        task_verificationComponents = task_verification.components
        for thisComponent in task_verification.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "task_verification" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        task_verification.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 7.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *task_verification_message* updates
            
            # if task_verification_message is starting this frame...
            if task_verification_message.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                task_verification_message.frameNStart = frameN  # exact frame index
                task_verification_message.tStart = t  # local t and not account for scr refresh
                task_verification_message.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(task_verification_message, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'task_verification_message.started')
                # update status
                task_verification_message.status = STARTED
                task_verification_message.setAutoDraw(True)
            
            # if task_verification_message is active this frame...
            if task_verification_message.status == STARTED:
                # update params
                task_verification_message.setColor(color_text, colorSpace='rgb', log=False)
            
            # if task_verification_message is stopping this frame...
            if task_verification_message.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > task_verification_message.tStartRefresh + 7-frameTolerance:
                    # keep track of stop time/frame for later
                    task_verification_message.tStop = t  # not accounting for scr refresh
                    task_verification_message.tStopRefresh = tThisFlipGlobal  # on global time
                    task_verification_message.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'task_verification_message.stopped')
                    # update status
                    task_verification_message.status = FINISHED
                    task_verification_message.setAutoDraw(False)
            # Run 'Each Frame' code from code
            keys = event.getKeys(keyList=['1', '2'])
            
            if '1' in keys:
                color_text = 'green'
            elif '2' in keys:
                color_text = 'red'
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                task_verification.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in task_verification.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "task_verification" ---
        for thisComponent in task_verification.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for task_verification
        task_verification.tStop = globalClock.getTime(format='float')
        task_verification.tStopRefresh = tThisFlipGlobal
        thisExp.addData('task_verification.stopped', task_verification.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if task_verification.maxDurationReached:
            routineTimer.addTime(-task_verification.maxDuration)
        elif task_verification.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-7.000000)
        
        # --- Prepare to start Routine "control" ---
        # create an object to store info about Routine control
        control = data.Routine(
            name='control',
            components=[textbox],
        )
        control.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        textbox.reset()
        # store start times for control
        control.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        control.tStart = globalClock.getTime(format='float')
        control.status = STARTED
        thisExp.addData('control.started', control.tStart)
        control.maxDuration = None
        # keep track of which components have finished
        controlComponents = control.components
        for thisComponent in control.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "control" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        control.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 20.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *textbox* updates
            
            # if textbox is starting this frame...
            if textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textbox.frameNStart = frameN  # exact frame index
                textbox.tStart = t  # local t and not account for scr refresh
                textbox.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textbox, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textbox.started')
                # update status
                textbox.status = STARTED
                textbox.setAutoDraw(True)
            
            # if textbox is active this frame...
            if textbox.status == STARTED:
                # update params
                textbox.setText(AI_text_01
                , log=False)
            
            # if textbox is stopping this frame...
            if textbox.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > textbox.tStartRefresh + 20-frameTolerance:
                    # keep track of stop time/frame for later
                    textbox.tStop = t  # not accounting for scr refresh
                    textbox.tStopRefresh = tThisFlipGlobal  # on global time
                    textbox.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'textbox.stopped')
                    # update status
                    textbox.status = FINISHED
                    textbox.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                control.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in control.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "control" ---
        for thisComponent in control.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for control
        control.tStop = globalClock.getTime(format='float')
        control.tStopRefresh = tThisFlipGlobal
        thisExp.addData('control.stopped', control.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if control.maxDurationReached:
            routineTimer.addTime(-control.maxDuration)
        elif control.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-20.000000)
        
        # --- Prepare to start Routine "control_verification" ---
        # create an object to store info about Routine control_verification
        control_verification = data.Routine(
            name='control_verification',
            components=[text],
        )
        control_verification.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_2
        color_text = 'white'  # default
        # store start times for control_verification
        control_verification.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        control_verification.tStart = globalClock.getTime(format='float')
        control_verification.status = STARTED
        thisExp.addData('control_verification.started', control_verification.tStart)
        control_verification.maxDuration = None
        # keep track of which components have finished
        control_verificationComponents = control_verification.components
        for thisComponent in control_verification.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "control_verification" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        control_verification.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 7.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text* updates
            
            # if text is starting this frame...
            if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text.frameNStart = frameN  # exact frame index
                text.tStart = t  # local t and not account for scr refresh
                text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.started')
                # update status
                text.status = STARTED
                text.setAutoDraw(True)
            
            # if text is active this frame...
            if text.status == STARTED:
                # update params
                text.setColor(color_text, colorSpace='rgb', log=False)
            
            # if text is stopping this frame...
            if text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text.tStartRefresh + 7-frameTolerance:
                    # keep track of stop time/frame for later
                    text.tStop = t  # not accounting for scr refresh
                    text.tStopRefresh = tThisFlipGlobal  # on global time
                    text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text.stopped')
                    # update status
                    text.status = FINISHED
                    text.setAutoDraw(False)
            # Run 'Each Frame' code from code_2
            keys = event.getKeys(keyList=['1', '2'])
            
            if '1' in keys:
                color_text = 'green'
            elif '2' in keys:
                color_text = 'red'
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                control_verification.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in control_verification.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "control_verification" ---
        for thisComponent in control_verification.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for control_verification
        control_verification.tStop = globalClock.getTime(format='float')
        control_verification.tStopRefresh = tThisFlipGlobal
        thisExp.addData('control_verification.stopped', control_verification.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if control_verification.maxDurationReached:
            routineTimer.addTime(-control_verification.maxDuration)
        elif control_verification.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-7.000000)
        
        # --- Prepare to start Routine "rest" ---
        # create an object to store info about Routine rest
        rest = data.Routine(
            name='rest',
            components=[rest_fixation],
        )
        rest.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for rest
        rest.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        rest.tStart = globalClock.getTime(format='float')
        rest.status = STARTED
        thisExp.addData('rest.started', rest.tStart)
        rest.maxDuration = None
        # keep track of which components have finished
        restComponents = rest.components
        for thisComponent in rest.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "rest" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        rest.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 12.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *rest_fixation* updates
            
            # if rest_fixation is starting this frame...
            if rest_fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                rest_fixation.frameNStart = frameN  # exact frame index
                rest_fixation.tStart = t  # local t and not account for scr refresh
                rest_fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rest_fixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rest_fixation.started')
                # update status
                rest_fixation.status = STARTED
                rest_fixation.setAutoDraw(True)
            
            # if rest_fixation is active this frame...
            if rest_fixation.status == STARTED:
                # update params
                pass
            
            # if rest_fixation is stopping this frame...
            if rest_fixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > rest_fixation.tStartRefresh + 12-frameTolerance:
                    # keep track of stop time/frame for later
                    rest_fixation.tStop = t  # not accounting for scr refresh
                    rest_fixation.tStopRefresh = tThisFlipGlobal  # on global time
                    rest_fixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rest_fixation.stopped')
                    # update status
                    rest_fixation.status = FINISHED
                    rest_fixation.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                rest.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in rest.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "rest" ---
        for thisComponent in rest.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for rest
        rest.tStop = globalClock.getTime(format='float')
        rest.tStopRefresh = tThisFlipGlobal
        thisExp.addData('rest.stopped', rest.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if rest.maxDurationReached:
            routineTimer.addTime(-rest.maxDuration)
        elif rest.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-12.000000)
        
        # --- Prepare to start Routine "reading2" ---
        # create an object to store info about Routine reading2
        reading2 = data.Routine(
            name='reading2',
            components=[reading2_textbox],
        )
        reading2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        reading2_textbox.reset()
        # store start times for reading2
        reading2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        reading2.tStart = globalClock.getTime(format='float')
        reading2.status = STARTED
        thisExp.addData('reading2.started', reading2.tStart)
        reading2.maxDuration = None
        # keep track of which components have finished
        reading2Components = reading2.components
        for thisComponent in reading2.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "reading2" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        reading2.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 10.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *reading2_textbox* updates
            
            # if reading2_textbox is starting this frame...
            if reading2_textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                reading2_textbox.frameNStart = frameN  # exact frame index
                reading2_textbox.tStart = t  # local t and not account for scr refresh
                reading2_textbox.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(reading2_textbox, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'reading2_textbox.started')
                # update status
                reading2_textbox.status = STARTED
                reading2_textbox.setAutoDraw(True)
            
            # if reading2_textbox is active this frame...
            if reading2_textbox.status == STARTED:
                # update params
                reading2_textbox.setText(AI_text_02, log=False)
            
            # if reading2_textbox is stopping this frame...
            if reading2_textbox.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > reading2_textbox.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    reading2_textbox.tStop = t  # not accounting for scr refresh
                    reading2_textbox.tStopRefresh = tThisFlipGlobal  # on global time
                    reading2_textbox.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'reading2_textbox.stopped')
                    # update status
                    reading2_textbox.status = FINISHED
                    reading2_textbox.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                reading2.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in reading2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "reading2" ---
        for thisComponent in reading2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for reading2
        reading2.tStop = globalClock.getTime(format='float')
        reading2.tStopRefresh = tThisFlipGlobal
        thisExp.addData('reading2.stopped', reading2.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if reading2.maxDurationReached:
            routineTimer.addTime(-reading2.maxDuration)
        elif reading2.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-10.000000)
        
        # --- Prepare to start Routine "task2" ---
        # create an object to store info about Routine task2
        task2 = data.Routine(
            name='task2',
            components=[task2_textbox],
        )
        task2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        task2_textbox.reset()
        # store start times for task2
        task2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        task2.tStart = globalClock.getTime(format='float')
        task2.status = STARTED
        thisExp.addData('task2.started', task2.tStart)
        task2.maxDuration = None
        # keep track of which components have finished
        task2Components = task2.components
        for thisComponent in task2.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "task2" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        task2.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 20.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *task2_textbox* updates
            
            # if task2_textbox is starting this frame...
            if task2_textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                task2_textbox.frameNStart = frameN  # exact frame index
                task2_textbox.tStart = t  # local t and not account for scr refresh
                task2_textbox.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(task2_textbox, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'task2_textbox.started')
                # update status
                task2_textbox.status = STARTED
                task2_textbox.setAutoDraw(True)
            
            # if task2_textbox is active this frame...
            if task2_textbox.status == STARTED:
                # update params
                task2_textbox.setText(AI_text_02, log=False)
            
            # if task2_textbox is stopping this frame...
            if task2_textbox.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > task2_textbox.tStartRefresh + 20-frameTolerance:
                    # keep track of stop time/frame for later
                    task2_textbox.tStop = t  # not accounting for scr refresh
                    task2_textbox.tStopRefresh = tThisFlipGlobal  # on global time
                    task2_textbox.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'task2_textbox.stopped')
                    # update status
                    task2_textbox.status = FINISHED
                    task2_textbox.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                task2.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in task2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "task2" ---
        for thisComponent in task2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for task2
        task2.tStop = globalClock.getTime(format='float')
        task2.tStopRefresh = tThisFlipGlobal
        thisExp.addData('task2.stopped', task2.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if task2.maxDurationReached:
            routineTimer.addTime(-task2.maxDuration)
        elif task2.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-20.000000)
        
        # --- Prepare to start Routine "task_verification" ---
        # create an object to store info about Routine task_verification
        task_verification = data.Routine(
            name='task_verification',
            components=[task_verification_message],
        )
        task_verification.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code
        color_text = 'white'  # default
        # store start times for task_verification
        task_verification.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        task_verification.tStart = globalClock.getTime(format='float')
        task_verification.status = STARTED
        thisExp.addData('task_verification.started', task_verification.tStart)
        task_verification.maxDuration = None
        # keep track of which components have finished
        task_verificationComponents = task_verification.components
        for thisComponent in task_verification.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "task_verification" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        task_verification.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 7.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *task_verification_message* updates
            
            # if task_verification_message is starting this frame...
            if task_verification_message.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                task_verification_message.frameNStart = frameN  # exact frame index
                task_verification_message.tStart = t  # local t and not account for scr refresh
                task_verification_message.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(task_verification_message, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'task_verification_message.started')
                # update status
                task_verification_message.status = STARTED
                task_verification_message.setAutoDraw(True)
            
            # if task_verification_message is active this frame...
            if task_verification_message.status == STARTED:
                # update params
                task_verification_message.setColor(color_text, colorSpace='rgb', log=False)
            
            # if task_verification_message is stopping this frame...
            if task_verification_message.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > task_verification_message.tStartRefresh + 7-frameTolerance:
                    # keep track of stop time/frame for later
                    task_verification_message.tStop = t  # not accounting for scr refresh
                    task_verification_message.tStopRefresh = tThisFlipGlobal  # on global time
                    task_verification_message.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'task_verification_message.stopped')
                    # update status
                    task_verification_message.status = FINISHED
                    task_verification_message.setAutoDraw(False)
            # Run 'Each Frame' code from code
            keys = event.getKeys(keyList=['1', '2'])
            
            if '1' in keys:
                color_text = 'green'
            elif '2' in keys:
                color_text = 'red'
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                task_verification.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in task_verification.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "task_verification" ---
        for thisComponent in task_verification.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for task_verification
        task_verification.tStop = globalClock.getTime(format='float')
        task_verification.tStopRefresh = tThisFlipGlobal
        thisExp.addData('task_verification.stopped', task_verification.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if task_verification.maxDurationReached:
            routineTimer.addTime(-task_verification.maxDuration)
        elif task_verification.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-7.000000)
        
        # --- Prepare to start Routine "control2" ---
        # create an object to store info about Routine control2
        control2 = data.Routine(
            name='control2',
            components=[control2_textbox],
        )
        control2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        control2_textbox.reset()
        # store start times for control2
        control2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        control2.tStart = globalClock.getTime(format='float')
        control2.status = STARTED
        thisExp.addData('control2.started', control2.tStart)
        control2.maxDuration = None
        # keep track of which components have finished
        control2Components = control2.components
        for thisComponent in control2.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "control2" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        control2.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 20.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *control2_textbox* updates
            
            # if control2_textbox is starting this frame...
            if control2_textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                control2_textbox.frameNStart = frameN  # exact frame index
                control2_textbox.tStart = t  # local t and not account for scr refresh
                control2_textbox.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(control2_textbox, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'control2_textbox.started')
                # update status
                control2_textbox.status = STARTED
                control2_textbox.setAutoDraw(True)
            
            # if control2_textbox is active this frame...
            if control2_textbox.status == STARTED:
                # update params
                control2_textbox.setText(AI_text_02, log=False)
            
            # if control2_textbox is stopping this frame...
            if control2_textbox.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > control2_textbox.tStartRefresh + 20-frameTolerance:
                    # keep track of stop time/frame for later
                    control2_textbox.tStop = t  # not accounting for scr refresh
                    control2_textbox.tStopRefresh = tThisFlipGlobal  # on global time
                    control2_textbox.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'control2_textbox.stopped')
                    # update status
                    control2_textbox.status = FINISHED
                    control2_textbox.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                control2.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in control2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "control2" ---
        for thisComponent in control2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for control2
        control2.tStop = globalClock.getTime(format='float')
        control2.tStopRefresh = tThisFlipGlobal
        thisExp.addData('control2.stopped', control2.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if control2.maxDurationReached:
            routineTimer.addTime(-control2.maxDuration)
        elif control2.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-20.000000)
        
        # --- Prepare to start Routine "control_verification" ---
        # create an object to store info about Routine control_verification
        control_verification = data.Routine(
            name='control_verification',
            components=[text],
        )
        control_verification.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_2
        color_text = 'white'  # default
        # store start times for control_verification
        control_verification.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        control_verification.tStart = globalClock.getTime(format='float')
        control_verification.status = STARTED
        thisExp.addData('control_verification.started', control_verification.tStart)
        control_verification.maxDuration = None
        # keep track of which components have finished
        control_verificationComponents = control_verification.components
        for thisComponent in control_verification.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "control_verification" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        control_verification.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 7.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text* updates
            
            # if text is starting this frame...
            if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text.frameNStart = frameN  # exact frame index
                text.tStart = t  # local t and not account for scr refresh
                text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.started')
                # update status
                text.status = STARTED
                text.setAutoDraw(True)
            
            # if text is active this frame...
            if text.status == STARTED:
                # update params
                text.setColor(color_text, colorSpace='rgb', log=False)
            
            # if text is stopping this frame...
            if text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text.tStartRefresh + 7-frameTolerance:
                    # keep track of stop time/frame for later
                    text.tStop = t  # not accounting for scr refresh
                    text.tStopRefresh = tThisFlipGlobal  # on global time
                    text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text.stopped')
                    # update status
                    text.status = FINISHED
                    text.setAutoDraw(False)
            # Run 'Each Frame' code from code_2
            keys = event.getKeys(keyList=['1', '2'])
            
            if '1' in keys:
                color_text = 'green'
            elif '2' in keys:
                color_text = 'red'
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                control_verification.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in control_verification.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "control_verification" ---
        for thisComponent in control_verification.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for control_verification
        control_verification.tStop = globalClock.getTime(format='float')
        control_verification.tStopRefresh = tThisFlipGlobal
        thisExp.addData('control_verification.stopped', control_verification.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if control_verification.maxDurationReached:
            routineTimer.addTime(-control_verification.maxDuration)
        elif control_verification.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-7.000000)
        
        # --- Prepare to start Routine "rest" ---
        # create an object to store info about Routine rest
        rest = data.Routine(
            name='rest',
            components=[rest_fixation],
        )
        rest.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for rest
        rest.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        rest.tStart = globalClock.getTime(format='float')
        rest.status = STARTED
        thisExp.addData('rest.started', rest.tStart)
        rest.maxDuration = None
        # keep track of which components have finished
        restComponents = rest.components
        for thisComponent in rest.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "rest" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        rest.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 12.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *rest_fixation* updates
            
            # if rest_fixation is starting this frame...
            if rest_fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                rest_fixation.frameNStart = frameN  # exact frame index
                rest_fixation.tStart = t  # local t and not account for scr refresh
                rest_fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rest_fixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rest_fixation.started')
                # update status
                rest_fixation.status = STARTED
                rest_fixation.setAutoDraw(True)
            
            # if rest_fixation is active this frame...
            if rest_fixation.status == STARTED:
                # update params
                pass
            
            # if rest_fixation is stopping this frame...
            if rest_fixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > rest_fixation.tStartRefresh + 12-frameTolerance:
                    # keep track of stop time/frame for later
                    rest_fixation.tStop = t  # not accounting for scr refresh
                    rest_fixation.tStopRefresh = tThisFlipGlobal  # on global time
                    rest_fixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rest_fixation.stopped')
                    # update status
                    rest_fixation.status = FINISHED
                    rest_fixation.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                rest.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in rest.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "rest" ---
        for thisComponent in rest.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for rest
        rest.tStop = globalClock.getTime(format='float')
        rest.tStopRefresh = tThisFlipGlobal
        thisExp.addData('rest.stopped', rest.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if rest.maxDurationReached:
            routineTimer.addTime(-rest.maxDuration)
        elif rest.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-12.000000)
        
        # --- Prepare to start Routine "reading3" ---
        # create an object to store info about Routine reading3
        reading3 = data.Routine(
            name='reading3',
            components=[reading3_textbox],
        )
        reading3.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        reading3_textbox.reset()
        # store start times for reading3
        reading3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        reading3.tStart = globalClock.getTime(format='float')
        reading3.status = STARTED
        thisExp.addData('reading3.started', reading3.tStart)
        reading3.maxDuration = None
        # keep track of which components have finished
        reading3Components = reading3.components
        for thisComponent in reading3.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "reading3" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        reading3.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 10.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *reading3_textbox* updates
            
            # if reading3_textbox is starting this frame...
            if reading3_textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                reading3_textbox.frameNStart = frameN  # exact frame index
                reading3_textbox.tStart = t  # local t and not account for scr refresh
                reading3_textbox.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(reading3_textbox, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'reading3_textbox.started')
                # update status
                reading3_textbox.status = STARTED
                reading3_textbox.setAutoDraw(True)
            
            # if reading3_textbox is active this frame...
            if reading3_textbox.status == STARTED:
                # update params
                reading3_textbox.setText(AI_text_03, log=False)
            
            # if reading3_textbox is stopping this frame...
            if reading3_textbox.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > reading3_textbox.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    reading3_textbox.tStop = t  # not accounting for scr refresh
                    reading3_textbox.tStopRefresh = tThisFlipGlobal  # on global time
                    reading3_textbox.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'reading3_textbox.stopped')
                    # update status
                    reading3_textbox.status = FINISHED
                    reading3_textbox.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                reading3.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in reading3.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "reading3" ---
        for thisComponent in reading3.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for reading3
        reading3.tStop = globalClock.getTime(format='float')
        reading3.tStopRefresh = tThisFlipGlobal
        thisExp.addData('reading3.stopped', reading3.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if reading3.maxDurationReached:
            routineTimer.addTime(-reading3.maxDuration)
        elif reading3.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-10.000000)
        
        # --- Prepare to start Routine "task3" ---
        # create an object to store info about Routine task3
        task3 = data.Routine(
            name='task3',
            components=[task3_textbox],
        )
        task3.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        task3_textbox.reset()
        # store start times for task3
        task3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        task3.tStart = globalClock.getTime(format='float')
        task3.status = STARTED
        thisExp.addData('task3.started', task3.tStart)
        task3.maxDuration = None
        # keep track of which components have finished
        task3Components = task3.components
        for thisComponent in task3.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "task3" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        task3.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 20.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *task3_textbox* updates
            
            # if task3_textbox is starting this frame...
            if task3_textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                task3_textbox.frameNStart = frameN  # exact frame index
                task3_textbox.tStart = t  # local t and not account for scr refresh
                task3_textbox.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(task3_textbox, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'task3_textbox.started')
                # update status
                task3_textbox.status = STARTED
                task3_textbox.setAutoDraw(True)
            
            # if task3_textbox is active this frame...
            if task3_textbox.status == STARTED:
                # update params
                task3_textbox.setText(AI_text_03, log=False)
            
            # if task3_textbox is stopping this frame...
            if task3_textbox.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > task3_textbox.tStartRefresh + 20-frameTolerance:
                    # keep track of stop time/frame for later
                    task3_textbox.tStop = t  # not accounting for scr refresh
                    task3_textbox.tStopRefresh = tThisFlipGlobal  # on global time
                    task3_textbox.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'task3_textbox.stopped')
                    # update status
                    task3_textbox.status = FINISHED
                    task3_textbox.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                task3.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in task3.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "task3" ---
        for thisComponent in task3.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for task3
        task3.tStop = globalClock.getTime(format='float')
        task3.tStopRefresh = tThisFlipGlobal
        thisExp.addData('task3.stopped', task3.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if task3.maxDurationReached:
            routineTimer.addTime(-task3.maxDuration)
        elif task3.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-20.000000)
        
        # --- Prepare to start Routine "task_verification" ---
        # create an object to store info about Routine task_verification
        task_verification = data.Routine(
            name='task_verification',
            components=[task_verification_message],
        )
        task_verification.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code
        color_text = 'white'  # default
        # store start times for task_verification
        task_verification.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        task_verification.tStart = globalClock.getTime(format='float')
        task_verification.status = STARTED
        thisExp.addData('task_verification.started', task_verification.tStart)
        task_verification.maxDuration = None
        # keep track of which components have finished
        task_verificationComponents = task_verification.components
        for thisComponent in task_verification.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "task_verification" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        task_verification.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 7.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *task_verification_message* updates
            
            # if task_verification_message is starting this frame...
            if task_verification_message.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                task_verification_message.frameNStart = frameN  # exact frame index
                task_verification_message.tStart = t  # local t and not account for scr refresh
                task_verification_message.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(task_verification_message, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'task_verification_message.started')
                # update status
                task_verification_message.status = STARTED
                task_verification_message.setAutoDraw(True)
            
            # if task_verification_message is active this frame...
            if task_verification_message.status == STARTED:
                # update params
                task_verification_message.setColor(color_text, colorSpace='rgb', log=False)
            
            # if task_verification_message is stopping this frame...
            if task_verification_message.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > task_verification_message.tStartRefresh + 7-frameTolerance:
                    # keep track of stop time/frame for later
                    task_verification_message.tStop = t  # not accounting for scr refresh
                    task_verification_message.tStopRefresh = tThisFlipGlobal  # on global time
                    task_verification_message.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'task_verification_message.stopped')
                    # update status
                    task_verification_message.status = FINISHED
                    task_verification_message.setAutoDraw(False)
            # Run 'Each Frame' code from code
            keys = event.getKeys(keyList=['1', '2'])
            
            if '1' in keys:
                color_text = 'green'
            elif '2' in keys:
                color_text = 'red'
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                task_verification.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in task_verification.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "task_verification" ---
        for thisComponent in task_verification.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for task_verification
        task_verification.tStop = globalClock.getTime(format='float')
        task_verification.tStopRefresh = tThisFlipGlobal
        thisExp.addData('task_verification.stopped', task_verification.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if task_verification.maxDurationReached:
            routineTimer.addTime(-task_verification.maxDuration)
        elif task_verification.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-7.000000)
        
        # --- Prepare to start Routine "control3" ---
        # create an object to store info about Routine control3
        control3 = data.Routine(
            name='control3',
            components=[control3_textbox],
        )
        control3.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        control3_textbox.reset()
        # store start times for control3
        control3.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        control3.tStart = globalClock.getTime(format='float')
        control3.status = STARTED
        thisExp.addData('control3.started', control3.tStart)
        control3.maxDuration = None
        # keep track of which components have finished
        control3Components = control3.components
        for thisComponent in control3.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "control3" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        control3.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 20.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *control3_textbox* updates
            
            # if control3_textbox is starting this frame...
            if control3_textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                control3_textbox.frameNStart = frameN  # exact frame index
                control3_textbox.tStart = t  # local t and not account for scr refresh
                control3_textbox.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(control3_textbox, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'control3_textbox.started')
                # update status
                control3_textbox.status = STARTED
                control3_textbox.setAutoDraw(True)
            
            # if control3_textbox is active this frame...
            if control3_textbox.status == STARTED:
                # update params
                control3_textbox.setText(AI_text_03, log=False)
            
            # if control3_textbox is stopping this frame...
            if control3_textbox.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > control3_textbox.tStartRefresh + 20-frameTolerance:
                    # keep track of stop time/frame for later
                    control3_textbox.tStop = t  # not accounting for scr refresh
                    control3_textbox.tStopRefresh = tThisFlipGlobal  # on global time
                    control3_textbox.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'control3_textbox.stopped')
                    # update status
                    control3_textbox.status = FINISHED
                    control3_textbox.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                control3.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in control3.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "control3" ---
        for thisComponent in control3.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for control3
        control3.tStop = globalClock.getTime(format='float')
        control3.tStopRefresh = tThisFlipGlobal
        thisExp.addData('control3.stopped', control3.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if control3.maxDurationReached:
            routineTimer.addTime(-control3.maxDuration)
        elif control3.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-20.000000)
        
        # --- Prepare to start Routine "control_verification" ---
        # create an object to store info about Routine control_verification
        control_verification = data.Routine(
            name='control_verification',
            components=[text],
        )
        control_verification.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from code_2
        color_text = 'white'  # default
        # store start times for control_verification
        control_verification.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        control_verification.tStart = globalClock.getTime(format='float')
        control_verification.status = STARTED
        thisExp.addData('control_verification.started', control_verification.tStart)
        control_verification.maxDuration = None
        # keep track of which components have finished
        control_verificationComponents = control_verification.components
        for thisComponent in control_verification.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "control_verification" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        control_verification.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 7.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text* updates
            
            # if text is starting this frame...
            if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text.frameNStart = frameN  # exact frame index
                text.tStart = t  # local t and not account for scr refresh
                text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.started')
                # update status
                text.status = STARTED
                text.setAutoDraw(True)
            
            # if text is active this frame...
            if text.status == STARTED:
                # update params
                text.setColor(color_text, colorSpace='rgb', log=False)
            
            # if text is stopping this frame...
            if text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text.tStartRefresh + 7-frameTolerance:
                    # keep track of stop time/frame for later
                    text.tStop = t  # not accounting for scr refresh
                    text.tStopRefresh = tThisFlipGlobal  # on global time
                    text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text.stopped')
                    # update status
                    text.status = FINISHED
                    text.setAutoDraw(False)
            # Run 'Each Frame' code from code_2
            keys = event.getKeys(keyList=['1', '2'])
            
            if '1' in keys:
                color_text = 'green'
            elif '2' in keys:
                color_text = 'red'
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                control_verification.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in control_verification.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "control_verification" ---
        for thisComponent in control_verification.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for control_verification
        control_verification.tStop = globalClock.getTime(format='float')
        control_verification.tStopRefresh = tThisFlipGlobal
        thisExp.addData('control_verification.stopped', control_verification.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if control_verification.maxDurationReached:
            routineTimer.addTime(-control_verification.maxDuration)
        elif control_verification.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-7.000000)
        
        # --- Prepare to start Routine "rest" ---
        # create an object to store info about Routine rest
        rest = data.Routine(
            name='rest',
            components=[rest_fixation],
        )
        rest.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for rest
        rest.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        rest.tStart = globalClock.getTime(format='float')
        rest.status = STARTED
        thisExp.addData('rest.started', rest.tStart)
        rest.maxDuration = None
        # keep track of which components have finished
        restComponents = rest.components
        for thisComponent in rest.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "rest" ---
        # if trial has changed, end Routine now
        if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
            continueRoutine = False
        rest.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 12.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *rest_fixation* updates
            
            # if rest_fixation is starting this frame...
            if rest_fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                rest_fixation.frameNStart = frameN  # exact frame index
                rest_fixation.tStart = t  # local t and not account for scr refresh
                rest_fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rest_fixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rest_fixation.started')
                # update status
                rest_fixation.status = STARTED
                rest_fixation.setAutoDraw(True)
            
            # if rest_fixation is active this frame...
            if rest_fixation.status == STARTED:
                # update params
                pass
            
            # if rest_fixation is stopping this frame...
            if rest_fixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > rest_fixation.tStartRefresh + 12-frameTolerance:
                    # keep track of stop time/frame for later
                    rest_fixation.tStop = t  # not accounting for scr refresh
                    rest_fixation.tStopRefresh = tThisFlipGlobal  # on global time
                    rest_fixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'rest_fixation.stopped')
                    # update status
                    rest_fixation.status = FINISHED
                    rest_fixation.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                rest.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in rest.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "rest" ---
        for thisComponent in rest.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for rest
        rest.tStop = globalClock.getTime(format='float')
        rest.tStopRefresh = tThisFlipGlobal
        thisExp.addData('rest.stopped', rest.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if rest.maxDurationReached:
            routineTimer.addTime(-rest.maxDuration)
        elif rest.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-12.000000)
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "end" ---
    # create an object to store info about Routine end
    end = data.Routine(
        name='end',
        components=[end_message],
    )
    end.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for end
    end.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    end.tStart = globalClock.getTime(format='float')
    end.status = STARTED
    thisExp.addData('end.started', end.tStart)
    end.maxDuration = None
    # keep track of which components have finished
    endComponents = end.components
    for thisComponent in end.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "end" ---
    end.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 5.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *end_message* updates
        
        # if end_message is starting this frame...
        if end_message.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_message.frameNStart = frameN  # exact frame index
            end_message.tStart = t  # local t and not account for scr refresh
            end_message.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_message, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'end_message.started')
            # update status
            end_message.status = STARTED
            end_message.setAutoDraw(True)
        
        # if end_message is active this frame...
        if end_message.status == STARTED:
            # update params
            pass
        
        # if end_message is stopping this frame...
        if end_message.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > end_message.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                end_message.tStop = t  # not accounting for scr refresh
                end_message.tStopRefresh = tThisFlipGlobal  # on global time
                end_message.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'end_message.stopped')
                # update status
                end_message.status = FINISHED
                end_message.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            end.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in end.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end" ---
    for thisComponent in end.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for end
    end.tStop = globalClock.getTime(format='float')
    end.tStopRefresh = tThisFlipGlobal
    thisExp.addData('end.stopped', end.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if end.maxDurationReached:
        routineTimer.addTime(-end.maxDuration)
    elif end.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-5.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
