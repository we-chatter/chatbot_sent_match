# -*- coding: utf-8 -*- 

'''
Author: ByronVon
Email: wangy@craiditx.com
Version: 
Date: 2020-12-11 17:48:34
LastEditTime: 2020-12-14 00:02:39
'''

import logging


class Log(object):
    """自定义的logger"""
    def __init__(self, path='model.log', level='DEBUG'):
        self.__path = path
        self.__level = level
        self.__logger = logging.getLogger()
        self.__logger.setLevel(self.__level)

    def __init_handler(self):
        """初始化handler"""
        stream_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(self.__path, encoding='utf-8')
        return stream_handler, file_handler

    def __set_handler(self, stream_handler, file_handler, level='DEBUG'):
        """设置handler级别并添加到logger容器中"""
        stream_handler.setLevel(level)
        file_handler.setLevel(level)
        self.__logger.addHandler(stream_handler)
        self.__logger.addHandler(file_handler)

    def __set_formatter(self, stream_handler, file_handler):
        """设置日志的输出格式"""
        formatter = logging.Formatter('%(asctime)s--%(name)s--%(filename)s--%(levelname)s--%(message)s')
        stream_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

    def __close_handler(self, stream_handler, file_handler):
        """关闭handler"""
        stream_handler.close()
        file_handler.close()

    @property
    def Logger(self):
        """构造收集器，返回logger"""
        stream_handler, file_handler = self.__init_handler()
        self.__set_handler(stream_handler, file_handler)
        self.__set_formatter(stream_handler, file_handler)
        self.__close_handler(stream_handler, file_handler)
        return self.__logger