# !/usr/bin/env python3
# -*- coding:utf-8 -*-
#
# Author: FANG Junying, fangjunying@neuracle.cn
#
# Versions:
# 	v0.1: 2018-08-14, orignal
#   v1.0: 2020-04-27, add neuroscan parser
# Copyright (c) 2020 Neuracle, Inc. All Rights Reserved. http://neuracle.cn/

import socket
from struct import unpack
import numpy as np
from  threading import Lock, Thread, Event
import select,time

## create ringbuffer
class RingBuffer():
    def __init__(self,n_chan,n_points):
        self.n_chan = n_chan
        self.n_points = n_points
        self.buffer = np.zeros((n_chan, n_points))
        self.currentPtr = 0
        self.nUpdate = 0
    ## append buffer and update current pointer
    def appendBuffer(self,data):
        n = data.shape[1]
        self.buffer[:,np.mod(np.arange(self.currentPtr,self.currentPtr+n),self.n_points)] = data
        self.currentPtr =  np.mod(self.currentPtr+n-1, self.n_points) + 1
        self.nUpdate = self.nUpdate+n
    ## get data from buffer
    def getData(self):
        data = np.hstack([self.buffer[:,self.currentPtr:], self.buffer[:,:self.currentPtr]])
        return data
    # reset buffer
    def resetBuffer(self):
        self.buffer = np.zeros((self.n_chan, self.n_points))
        self.currentPtr = 0
        self.nUpdate = 0

## create a new thread used to receive data from Neuracle/DSI recorder software according TCP/IP socket
class DataServerThread(Thread,):
    sock = []
    _update_interval = 0.04  ## unit is seconds. dataserver sends TCP/IP socket in 40 milliseconds
    def __init__(self,device,n_chan,My_EEG_Data,srate=1000,t_buffer=3,):
        Thread.__init__(self)
        self.device = device
        self.n_chan = n_chan
        self.srate = srate
        self.t_buffer = t_buffer
        self.My_EEG_Data = My_EEG_Data


        # 显示时间窗口大小,40次采集的数据
        self.show_time = 0.05 * 1000 * 20  # ms 1s

        #计数变量
        self.init_cnt = 0
        self.cnt = 0
        self.step = 4 #步长为40*0.05  200ms

    def connect(self,hostname='127.0.0.1', port= 8712):
        """
        try to connect data server
        """
        self.hostname = hostname
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        notconnect = True
        reconnecttime = 0
        while notconnect:
            try:
                self.sock.connect((self.hostname, self.port))
                notconnect = False
            except:
                reconnecttime += 1
                print('connection failed, retrying for %d times' % reconnecttime)
                time.sleep(1)
                if reconnecttime > 2:
                    break
        self.shutdown_flag = Event()
        self.shutdown_flag.set()
        self.sock.setblocking(True)
        self.bufsize = int(self._update_interval*4*self.n_chan*self.srate*10)  # set buffer size
        nPoints= int(np.round(self.t_buffer*self.srate))
        self.ringBuffer = RingBuffer(self.n_chan, nPoints) # initiate the ringbuffer class
        self.buffer = b'' ## binary buffer used to collect binary array from data server
        return notconnect

    def run(self):
        self.read_thread()

    def read_thread(self): ## visit dataserver, catch sockets and parse sockets, append parsed data to ringbuffer
        socket_lock = Lock()
        while self.shutdown_flag.isSet():
            if not self.sock:
                break
            rs, _, _ = select.select([self.sock], [], [], 9)
            for r in rs:
                socket_lock.acquire()
                if not self.sock:
                    socket_lock.release()
                    break
                try:
                    raw = r.recv(self.bufsize)
                except:
                    print('can not recieve socket ...')
                    socket_lock.release()
                    self.sock.close()
                else:
                    raw = self.buffer + raw
                    data, evt = self.parseData(raw) ## parse data
                    socket_lock.release()
                    data = data.reshape(len(data) // (self.n_chan), self.n_chan)
                    self.ringBuffer.appendBuffer(data.T)

                    #直接接收数据
                    #初始化字典
                    if self.init_cnt == 0:
                        s = self.GetBufferData()
                        self.My_EEG_Data['eeg_data'] = s
                        self.My_EEG_Data['eeg_show_data'] = s
                        self.My_EEG_Data['eeg_all_data'] = s
                        self.My_EEG_Data['record_data'] = s
                        self.init_cnt = 1
                    #接收数据
                    else:
                        s = self.GetBufferData()
                        if (self.My_EEG_Data['eeg_show_data'][self.My_EEG_Data['select_chan']].size > self.show_time):
                            self.My_EEG_Data['eeg_show_data'] = np.delete(self.My_EEG_Data['eeg_show_data'],range(s[self.My_EEG_Data['select_chan']].size),axis=1)
                        self.My_EEG_Data['eeg_show_data'] = np.concatenate((self.My_EEG_Data['eeg_show_data'], s), axis=1)
                        self.My_EEG_Data['eeg_all_data'] = np.concatenate((self.My_EEG_Data['eeg_all_data'], s), axis=1)

                        if self.My_EEG_Data['start_flag'] == True:
                            self.My_EEG_Data['record_data'] = np.concatenate((self.My_EEG_Data['record_data'], s),
                                                                              axis=1)

                    #判断是否需要清空数据

                    if self.My_EEG_Data['clear_flag'] == 1 :
                        try:
                            self.My_EEG_Data['eeg_show_data'] = np.delete(self.My_EEG_Data['eeg_show_data'], range(
                                self.My_EEG_Data['eeg_show_data'][self.My_EEG_Data['select_chan']].size), axis=1)
                            self.My_EEG_Data['eeg_all_data'] = np.delete(self.My_EEG_Data['eeg_all_data'],
                                                                           range(self.My_EEG_Data['eeg_all_data'][self.My_EEG_Data['select_chan']].size),
                                                                           axis=1)
                            self.My_EEG_Data['record_data'] = np.delete(self.My_EEG_Data['record_data'],
                                                                         range(self.My_EEG_Data['record_data'][
                                                                                   self.My_EEG_Data[
                                                                                       'select_chan']].size),
                                                                         axis=1)
                            print('清除成功')
                            # self.My_EEG_Data['clear_flag'] = 0

                        except Exception as e:
                            print(e)
                            print( '清除失败')

                    self.cnt = self.cnt + 1
                    if self.cnt==4:
                        try:
                            self.My_EEG_Data['eeg_show_FFT'],self.My_EEG_Data['Fre_show'] = FFT(1000, self.My_EEG_Data['eeg_show_data'][self.My_EEG_Data['select_chan']])
                            self.My_EEG_Data['eeg_all_FFT'], self.My_EEG_Data['Fre_all'] = FFT(1000, self.My_EEG_Data[
                            'eeg_all_data'][self.My_EEG_Data['select_chan']])
                            self.cnt = 0
                        except:
                            print('wait data ...')
                        # show_data, n = FFT(1000, gloab_val.EEG_data['eeg_show_data'][0])


    def parseData(self,raw):
        if 'Neuracle' in self.device: ## parse data according to Neuracle device protocol
            n = len(raw)
            event , hexData  = [], []
            hexData = raw[:n - np.mod(n, 4 * self.n_chan)] # unpack hex-data  in row
            self.buffer = raw[n - np.mod(n, 4 * self.n_chan):]
            n_item = int(len(hexData)/4/self.n_chan)
            # format_str = '<' + (str(self.n_chan -1) + 'f' + '1I') * n_item
            format_str = '<' + (str(self.n_chan) + 'f') * n_item
            parse_data = unpack(format_str, hexData)

        elif 'DSI' in self.device : ## parse data according to DSI device protocol
            token = '@ABCD'
            n = len(raw)
            i = 0
            parse_data, data_record, event, event_record  = [], [], [], []
            iData = 0
            iEvent = 1
            while i + 12 < n:
                if token == raw[i:i + 5].decode('ascii'):
                    packetType = raw[i + 5]
                    # print(packetType)
                    bytenum = raw[i + 6:i + 8]
                    packetLength = 256 * bytenum[0] + bytenum[1]
                    # bytenum = unpack('>4I', buffer[i+8:i+12])
                    # packetNumber = 16777216*bytenum[0]+65536*bytenum[1]+256*bytenum[2]+bytenum[3]
                    if i + 12 + packetLength > n:
                        break
                    if packetType == 1:
                        data_record.append({})
                        # bytenum = unpack('>4I', buffer[i+12:i+16])
                        # data_record[iData]['TimeStamp'] = 16777216*bytenum[0]+65536*bytenum[1]+256*bytenum[2]+bytenum[3]
                        # data_record[iData]['DataCounter'] = unpack('>I', buffer[i+16])
                        # data_record[iData]['ADCStatus'] = unpack('>I', buffer[i+17:i+23])[0]
                        if np.mod(packetLength - 11, 4) != 0:
                            print('The packetLength may be incorrect!')
                        else:
                            pass
                        data_num = int((packetLength - 11) / 4)
                        format = '>' + str(data_num) + 'f'
                        data_record[iData]['ChannelData'] = unpack(format, raw[i + 23:i + 12 + packetLength])
                        parse_data.extend(data_record[iData]['ChannelData'])
                        iData += 1
                    elif packetType == 5:
                        event_record.append({})
                        # bytenum = unpack('>4I', buffer[i+12:i+16])
                        # event_record[iEvent]['EventCode'] = 16777216*bytenum[0]+65536*bytenum[1]+256*bytenum[2]+bytenum[3]
                        # bytenum = unpack('>4I', buffer[i+16:i+20])
                        # event_record[iEvent]['SendingNode'] = 16777216*bytenum[0]+65536*bytenum[1]+256*bytenum[2]+bytenum[3]
                        # if packerLength > 20:
                        #     bytenum = unpack('>4I', buffer[i+20:i+24])
                        #     event_record[iEvent]['MessageLength'] = 16777216*bytenum[0]+65536*bytenum[1]+256*bytenum[2]+bytenum[3]
                        #     event_record[iEvent]['Message'] = buffer[i+24:i+24+event[iEvent]['MessageLength']].decode('ascii')
                        # event.extend(event_record[iEvent]['Message'])
                        iEvent += 1
                    else:
                        pass
                    i = i + 12 + packetLength
                else:
                    i += 1
            self.buffer = raw[i:]
        elif 'Neuroscan' in self.device:
            data,event = [], []
            nHeader,nPoint, nBytes = 12, 40, 4
            nPacket = nHeader+nBytes*self.n_chan*nPoint
            n_item = 0
            while len(raw) >= nPacket:
                n_item = n_item + 1
                data.extend(raw[nHeader:nPacket])
                raw = raw[nPacket:]
            if len(data)>0:
                format = '>i'*self.n_chan*nPoint*n_item
                unpackData = np.asarray(unpack(format,data))
                n = len(unpackData)
                parse_data = unpackData.reshape(self.n_chan,n/self.n_chan)
                parse_data[:, :-1] = parse_data[:, :-1] * 0.14827
                parse_data[:,-1] =np.vstack((0,parse_data[1:,-1]-parse_data[:-1,-1]))
                parse_data = parse_data.flatten()
            else:
                print('Fetch data from NeuroScan failed .....')
                parse_data,evt = [],[]
        else:
            print('not avaliable device !')
            parse_data =[]
            event = []
            pass
        return np.asarray(parse_data), event

    ## get float data
    def GetBufferData(self):
        return self.ringBuffer.getData()

    # get current update point
    def GetDataLenCount(self):
        return self.ringBuffer.nUpdate

    # reset current update point
    def ResetDataLenCount(self, count=0):
        self.ringBuffer.nUpdate = count

    # reset trigger channel
    def ResetTriggerChanofBuff(self, data=None):
        if data == None:
            self.ringBuffer.buffer[-1, :] = np.zeros((1, self.ringBuffer.buffer.shape[-1]))
        else:
            self.ringBuffer.buffer[-1, :] = data

    # stop/close thread
    def stop(self):
        self.shutdown_flag.clear()