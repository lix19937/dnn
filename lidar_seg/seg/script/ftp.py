#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/11/24 10:32
# @Author  : 
# @Site    : 
# @File    : ftp.py
# @Software: PyCharm


import paramiko
import os
from stat import S_ISDIR as isdir
import time

class MyFtp:
    def __init__(self,username = "",password="",host="",port=22):
        self.username = username
        self.password = password
        self.host = host
        self.port = port

    #建立连接
    def sftp_connect(self):
        client = None
        sftp = None
        try:
            client = paramiko.Transport((self.host, self.port))
        except Exception as error:
            print(error)
        else:
            try:
                client.connect(username=self.username, password=self.password)
            except Exception as error:
                print(error)
            else:
                sftp = paramiko.SFTPClient.from_transport(client)
        return client,sftp

    #断开连接
    def sftp_disconnect(self,client):
        try:
            client.close()
        except Exception as error:
            print(error)

    #检查目录
    def _check_local(self,local):
        if not os.path.exists(local):
            try:
                os.mkdir(local)
            except IOError as err:
                print(err)

    #下载文件
    def sftp_downloadfile(self,sftp,local, remote):
        #检查远程文件是否存在
        try:
            result = sftp.stat(remote)
        except IOError as err:
            error = '[ERROR %s] %s: %s' %(err.errno,os.path.basename(os.path.normpath(remote)),err.strerror)
            print(error)
        else:
            #判断远程文件是否为目录
            if isdir(result.st_mode):
                # dirname = os.path.basename(os.path.normpath(remote))
                # local = os.path.join(local,dirname)
                self._check_local(local)
                for file in sftp.listdir(remote):
                    sub_remote = os.path.join(remote,file)
                    sub_remote = sub_remote.replace('\\','/')
                    self.sftp_downloadfile(sftp,local,sub_remote)
            else:
            #拷贝文件
                r_mtime = sftp.stat(remote).st_mtime#最后一次修改的时间
                r_file_modify_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(r_mtime))
                #print('远程文件修改时间：',r_file_modify_time)
                # local = remote
                if os.path.isdir(local):
                    local = os.path.join(local,os.path.basename(remote))
                #判断本地是否有该文件，没有则下载，有则判断文件最后修改时间是否一致，不一致则重新下载
                if os.path.exists(local):
                    l_mtime = os.stat(local).st_mtime#最后一次修改的时间
                    l_file_modify_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(l_mtime))
                    #print('本地文件修改时间：',l_file_modify_time)
                    if r_file_modify_time != l_file_modify_time :
                        sftp.get(remote,local)
                        print('[get]',local,'<==',remote)
                else:
                    sftp.get(remote,local)
                    print('[get]',local,'<==',remote)
    
    def sftp_upload_dir(self, sftp, local):
        remote = local
        # sftp = paramiko.SFTPClient.from_transport(client)
        if not os.path.exists(local):
            print('no local dir, please check the input!')
            return False
        try:
            sftp.stat(remote)
        except:
            sftp.mkdir(remote)
        else:
            print(f'remote {remote} already exists!')
            return False
        for file in os.listdir(local):
            sub_local = os.path.join(local, file)
            if os.path.isdir(sub_local):
                self.sftp_upload_dir(sftp, sub_local)
            else:
                sftp.put(sub_local, sub_local)
                print('[put]',sub_local,'==>',sub_local)
        return True



if __name__ == '__main__':
    remote = '/home/Linux-admin/gpulogger/log'

    #连接ftp，下载文件
    ip_list = ['10.94.63.17', '10.94.61.156', ]
    a100_name = ['SSASL3753', 'SSADL3780', ]
    for i, ip in enumerate(ip_list):
        ftp = MyFtp(username="e2pycharm",password="Pass12!@",host=ip,port=22)
        client,sftp = ftp.sftp_connect()
        local = '/home/igs/mnt/lidar_label/GPU_LOG/Lidar_detection/' + a100_name[i]
        ftp.sftp_downloadfile(sftp, local, remote)
        sftp.close()
