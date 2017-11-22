# This package will contain the spiders of your Scrapy project
#
# Please refer to the documentation for information on how to create and manage
# your spiders.
import scrapy
import re
import os
import sys
import json
import pickle
import yaml
import requests
import urllib2
import shutil
sys.path.insert(0, './spiders')
from urlparse import urlparse
from scrapy.utils.project import get_project_settings
from scrapy.selector import Selector
from url_constructor_file import url_constructor
GS_LINK_JSON_FILE       =  "output.json"
RESULT_FILE             = "htmlread_1.txt"
from fourth_year_project import settings
ENABLE_TEXT_SUMMARIZE   = 0 # For NLTK to look into the text for details.
ENABLE_PARAGRAPH_STORED = 1 # Store website content to file.
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
img_width, img_height = 150, 150
from keras.models import load_model
from keras import backend as K
K.set_image_dim_ordering('th')
class GoogleSpider(scrapy.Spider):
    # def __init__(self, conv_net_classify=None, *args, **kwargs):
    #     super(GoogleSpider, self).__init__(*args, **kwargs)
    #     self.conv_net_classify = conv_net_classify
    with open(RESULT_FILE,'w') as f:
        f.write('')
        print 'Restart the log file'
    with open(GS_LINK_JSON_FILE,'w') as f:
        f.write('')
        print 'Restart the GS_LINK_JSON_FILE file'
    search_class = url_constructor("car", "price")
    setting_data = search_class.retrieved_setting_fr_json_file()
    name = setting_data['Name']
    print "name"
    print name
    allowed_domains = setting_data['Domain']
    start_urls = setting_data['SearchUrl']
    print setting_data['type_of_parse']
    def combine_all_url_link_for_multiple_search(self,more_url_list):
        '''
            Combine all the url link list in the event of mutliple search.
            list more_url_list --> none
            get from Json file and eventually dump all back
        '''

        with open(GS_LINK_JSON_FILE, "r") as outfile:
            setting_data = yaml.load(outfile)
            if setting_data is None or not setting_data.has_key('output_url'):
                setting_data = dict()
                setting_data['output_url'] = []

        with open(GS_LINK_JSON_FILE, "w") as outfile:
            json.dump({'output_url': setting_data['output_url']+more_url_list}, outfile, indent=4)
    def remove_escape_characters(self, raw_input):
        for n in ['\n','\t','\r']:
            raw_input = raw_input.replace(n,'')
        return raw_input
    def join_list_of_str(self,list_of_str, joined_chars= '...'):
        return joined_chars.join([n for n in list_of_str])
    def parse(self, response):
        if self.setting_data['type_of_parse'] == 'google_search':
            print 'For google search parsing'

            ## Get the selector for xpath parsing
            sel = Selector(response)
            print("SELECTOR")
            print sel
            google_search_links_list =  sel.xpath('//h3/a/@href').extract()
            google_search_links_list = [re.search('q=(.*)&sa',n).group(1) for n in google_search_links_list if re.search('q=(.*)&sa',n)]
            print google_search_links_list
            print len(google_search_links_list)
            ## Display a list of the result link
            for n in google_search_links_list:
                print n

            self.combine_all_url_link_for_multiple_search(google_search_links_list)
        if self.setting_data['type_of_parse'] == 'general':
            # Need to separate out one subject - i.e. need to find price for Toyota SUV 1000 or whatever, Chevrolet 3000, etc. Give some example car brands.
            # Need to specify purpose - image detection, label (e.g. price prediction) and datatype. Find label and
            # Also need to include false data.
            print
            print "General link scraping"
            sel = Selector(response)
            print sel
            title = sel.xpath('//title/text()').extract()
            if len(title)>0:
                title = title[0].encode(errors='replace') #replace any unknown character with ?
            contents = sel.xpath('/html/head/meta[@name="description"]/@content').extract()
            if len(contents)>0:
                contents = contents[0].encode(errors='replace') #replace any unknown character with ?
            paragraph_list = sel.xpath('//p/text()').extract()
            para_str = self.join_list_of_str(paragraph_list, joined_chars= '..')
            para_str = para_str.encode(errors='replace')
            para_str = self.remove_escape_characters(para_str)
            print
            print title
            print
            print contents
            ## Dump results to text file
            with open(RESULT_FILE,'a') as f:
                f.write('\n')
                f.write('#'*20)
                f.write('\n')
                f.write(title + '\n')
                f.write(response.url)
                for n in range(2): f.write('\n')
                f.write(str(contents))
                for n in range(2): f.write('\n')
                f.write(para_str)
                f.write('\n')
                f.write('#'*20)
                for n in range(2): f.write('\n')

            print
            print 'Completed'
        if self.setting_data['type_of_parse'] == 'image':
            header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"
            }
            sel = Selector(response)
            parsed_uri = urlparse(response.url)
            domain = '{uri.scheme}://{uri.netloc}/'.format(uri=parsed_uri)
            image_array = sel.xpath('//img[contains(@src, "cat")]/@src').extract()
            for i, link in enumerate(image_array):
                if link.startswith("http"):
                    pass
                else:
                    image_array[i] = domain + link

            link_array = sel.xpath('//a[contains(@href, "cat")]//@href').extract()
            plural_link_array = sel.xpath('//a[contains(@href, "cats")]//@href').extract()
            link_array = link_array + plural_link_array
            for i, link in enumerate(link_array):
                if link.startswith("http"):
                    pass
                else:
                    link_array[i] = domain + link
            new_url_array = []
            already_downloaded = []
            with open("cat_scrape_link.json", "r") as outfile:
                if os.stat("cat_scrape_link.json").st_size == 0:
                    new_url_array = link_array
                else:
                    json_data = json.load(outfile)
                    output_url_array =  json_data['output_url']
                    #HERE change here
                    new_url_array = list(set(output_url_array + link_array))
            with open("cat_scraped_images.json", "r") as outfile:
                if os.stat("cat_scraped_images.json").st_size != 0:
                    # read scrape url
                    scraped_json_data = json.load(outfile)
                    already_downloaded = scraped_json_data['scraped_urls']
            with open('cat_scrape_link.json', "w") as outfile:
                json.dump({'output_url': new_url_array}, outfile, indent=4)

            unlabelled_img_dir = 'unlabelled_image_sets/cat'
            labelled_img_dir = './image_sets/cat'
            print("CONV NET CLASSIFY", self.setting_data['conv_net_classify'])
            if self.setting_data['conv_net_classify'] != True:
                for index,img in enumerate(image_array):
                    #handle if there are too many images
                    if len(os.listdir(unlabelled_img_dir)) < 1348:
                        if img not in already_downloaded:
                            try:
                                req = urllib2.Request(img,headers={'User-Agent': header})
                                raw_img = urllib2.urlopen(req).read()
                                counter = len([i for i in os.listdir(unlabelled_img_dir)]) + 1
                                print counter
                                print img[1]
                                if len(img[1]) == 0:
                                    f = open(os.path.join(unlabelled_img_dir, "image" + "_" + str(counter) + '.jpg'), 'wb')
                                else:
                                    if 'jpg' in img:
                                        f = open(os.path.join(unlabelled_img_dir, "image" + "_" + str(counter) + '.jpg'), 'wb')
                                    elif 'png' in img:
                                        f = open(os.path.join(unlabelled_img_dir, "image" + "_" + str(counter) + '.png'), 'wb')
                                    else:
                                        f = open(os.path.join(unlabelled_img_dir, "image" + "_" + str(counter) + '.jpg'), 'wb')
                                f.write(raw_img)
                                already_downloaded.append(img)
                                with open('cat_scraped_images.json', "w") as outfile   :
                                    print "already downloaded length"
                                    print len(already_downloaded)
                                    json.dump({'scraped_urls': already_downloaded}, outfile, indent=4)
                                f.close()

                                # CNN test shift unlabelled to labelled.
                            except Exception as e:
                                print "could not load : "+img
                                print e
            else:
                #use conv_net
              if len(os.listdir(labelled_img_dir)) < 3090:
                for index,img in enumerate(image_array):
                    if img not in already_downloaded:
                        download_path = './'
                        r = requests.get(img, stream = True, headers = header)
                        image_to_be_tested = ''
                        if r.status_code == 200:
                            full = r.headers.get('content-type')
                            ext = full.split('/')[1]
                            if ext == 'html' or ext == 'gif':
                              with open(os.path.join(download_path, "image_to_be_tested.jpg"), 'wb') as f:
                                  image_to_be_tested = os.path.join(download_path, "image_to_be_tested.jpg")
                                  r.raw.decode_content = True
                                  shutil.copyfileobj(r.raw, f)
                            else:
                              with open(os.path.join(download_path, "image_to_be_tested." + ext), 'wb') as f:
                                image_to_be_tested = os.path.join(download_path, "image_to_be_tested." + ext)
                                r.raw.decode_content = True
                                shutil.copyfileobj(r.raw, f)
                        K.set_image_dim_ordering('th')
                        img_width, img_height = 150, 150
                        model = load_model('open_set_cnn.h5')
                        image = load_img(image_to_be_tested,target_size=(img_width, img_height))
                        x = img_to_array(image)  # this is a Numpy array with shape (3, 150, 150)
                        x = x.reshape( (1,) + x.shape )  # this is a Numpy array with shape (1, 3, 150, 150)
                        result = model.predict(x)
                        result = list(result[0])
                        if result.index(max(result)) == 0:
                                try:
                                    print('cat one')
                                    download_path = './image_sets/cat'
                                    r = requests.get(img, stream = True)
                                    if r.status_code == 200:
                                        counter = len([i for i in os.listdir(download_path)]) + 1
                                        print counter
                                        full = r.headers.get('content-type')
                                        ext = full.split('/')[1]
                                        if ext == 'html' or ext == 'gif':
                                          with open(os.path.join(download_path, "image" + "_" + str(counter) + '.jpg'), 'wb') as f:
                                              r.raw.decode_content = True
                                              shutil.copyfileobj(r.raw, f)
                                        else:
                                          with open(os.path.join(download_path, "image" + "_" + str(counter) + '.' + ext), 'wb') as f:
                                            r.raw.decode_content = True
                                            shutil.copyfileobj(r.raw, f)
                                        already_downloaded.append(img)
                                        with open('cat_scraped_images.json', "w") as outfile:
                                            print "already downloaded length"
                                            print len(already_downloaded)
                                            json.dump({'scraped_urls': already_downloaded}, outfile, indent=4)
                                    # CNN test shift unlabelled to labelled.
                                except Exception as e:
                                    print "could not load : "+img
                                    print e
                        elif result.index(max(result)) == 1:
                                try:
                                    print('cat one')
                                    download_path = './image_sets/cat'
                                    r = requests.get(img, stream = True)
                                    if r.status_code == 200:
                                        counter = len([i for i in os.listdir(download_path)]) + 1
                                        print counter
                                        full = r.headers.get('content-type')
                                        ext = full.split('/')[1]
                                        if ext == 'html' or ext == 'gif':
                                          with open(os.path.join(download_path, "image" + "_" + str(counter) + '.jpg'), 'wb') as f:
                                              r.raw.decode_content = True
                                              shutil.copyfileobj(r.raw, f)
                                        else:
                                          with open(os.path.join(download_path, "image" + "_" + str(counter) + '.' + ext), 'wb') as f:
                                            r.raw.decode_content = True
                                            shutil.copyfileobj(r.raw, f)
                                        already_downloaded.append(img)
                                        with open('cat_scraped_images.json', "w") as outfile   :
                                            print "already downloaded length"
                                            print len(already_downloaded)
                                            json.dump({'scraped_urls': already_downloaded}, outfile, indent=4)

                                    # CNN test shift unlabelled to labelled.
                                except Exception as e:
                                    print "could not load : "+img
                                    print e
                        else:
                            try:
                                print('neither one')
                                download_path = './image_sets/neither'
                                r = requests.get(img, stream = True)
                                if r.status_code == 200:
                                    counter = len([i for i in os.listdir(download_path)]) + 1
                                    print counter
                                    full = r.headers.get('content-type')
                                    ext = full.split('/')[1]
                                    if ext == 'html' or ext == 'gif':
                                      with open(os.path.join(download_path, "neither_image" + "_" + str(counter) + '.jpg'), 'wb') as f:
                                          r.raw.decode_content = True
                                          shutil.copyfileobj(r.raw, f)
                                    else:
                                      with open(os.path.join(download_path, "neither_image" + "_" + str(counter) + '.' + ext), 'wb') as f:
                                        r.raw.decode_content = True
                                        shutil.copyfileobj(r.raw, f)
                                    already_downloaded.append(img)
                                    with open('neither_scraped_images.json', "w") as outfile   :
                                        print "already downloaded length"
                                        print len(already_downloaded)
                                        json.dump({'scraped_urls': already_downloaded}, outfile, indent=4)
                                # CNN test shift unlabelled to labelled.
                            except Exception as e:
                                print "could not load : "+img
                                print e


        # if len(os.listdir(unlabelled_img_dir)) < 1100:
        #     #scrape form images link scrape scrape_link
        #         print 'Start scraping more results'
        #         spider_file_path = '/Users/androswong/fourth_year_project/fourth_year_project/'
        #         spider_filename = 'images_link_scrape.py'
        #         ## Run the crawler -- and remove the pause if do not wish to see contents of the command prompt
        #         new_project_cmd = 'cd "%s" & python %s' %(spider_file_path,spider_filename)
        #         os.system(new_project_cmd)
