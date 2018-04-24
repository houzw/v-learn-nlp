#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: houzhiwei
# time: 2018/4/20 20:31


import scrapy


class BlogSpider(scrapy.Spider):
    name = 'blog'  # 爬虫的标识符

    def start_requests(self):
        """ must return an iterable of Requests 返回url的迭代器"""
        start_urls = ['http://quotes.toscrape.com/page/1/']
        for url in start_urls:
            yield scrapy.Request(url, self.parse)
        return super().start_requests()

    def parse(self, response):
        page = response.url.split('/')[-2]
        n = 'blog-%s.html' % page
        print(response.text)
        print(n)
