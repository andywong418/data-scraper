import re, os, sys, math, yaml
import json
sys.path.insert(0, './spiders')

from scrapy.spider import Spider
from scrapy.selector import Selector

from url_constructor_file import url_constructor
if __name__ == '__main__':
    print 'Start scrape individual results'
    search_subject= 'cat'
    search_label = ''
    #search_words = ['best area to stay in tokyo','cheap place to stay in tokyo']
    GS_LINK_JSON_FILE = search_subject + "_scrape_link.json" #must be same as the get_google_link_results.py
    # spider store location, depend on user input
    spider_file_path = '/Users/androswong/fourth_year_project/fourth_year_project/spiders'
    spider_filename = '__init__.py'
    crawler = url_constructor(search_subject, search_label)
    data  = crawler.retrieved_setting_fr_json_file(GS_LINK_JSON_FILE)

    #check if proper url --> must at least start with http
    url_links_image_search = [n for n in data['output_url'] if n.startswith('http')]

    ## Switch to the second seach
    crawler.data_format = 3

    crawler.url_array = url_links_image_search

    ## Set the setting for json
    temp_data_for_store = crawler.prepare_data_for_json_store()
    # print("TEMP DATA_FOR STORE", temp_data_for_store)
    temp_data_for_store['conv_net_classify'] = True
    crawler.set_setting_to_json_file(temp_data_for_store)

    ## Run the crawler -- and remove the pause if do not wish to see contents of the command prompt
    new_project_cmd = 'cd "%s" & scrapy runspider %s -a conv_net_classify=true' %(spider_file_path,spider_filename)
    os.system(new_project_cmd)
