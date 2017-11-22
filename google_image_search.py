from bs4 import BeautifulSoup
import requests
import re
import urllib2
import os
import cookielib
import json
import ast
import re
import tokenize
import token
from StringIO import StringIO
import sys

def fixLazyJsonWithComments (in_text):
  """ Same as fixLazyJson but removing comments as well
  """
  result = []
  tokengen = tokenize.generate_tokens(StringIO(in_text).readline)

  sline_comment = False
  mline_comment = False
  last_token = ''

  for tokid, tokval, _, _, _ in tokengen:

    # ignore single line and multi line comments
    if sline_comment:
      if (tokid == token.NEWLINE) or (tokid == tokenize.NL):
        sline_comment = False
      continue

    # ignore multi line comments
    if mline_comment:
      if (last_token == '*') and (tokval == '/'):
        mline_comment = False
      last_token = tokval
      continue

    # fix unquoted strings
    if (tokid == token.NAME):
      if tokval not in ['true', 'false', 'null', '-Infinity', 'Infinity', 'NaN']:
        tokid = token.STRING
        tokval = u'"%s"' % tokval

    # fix single-quoted strings
    elif (tokid == token.STRING):
      if tokval.startswith ("'"):
        tokval = u'"%s"' % tokval[1:-1].replace ('"', '\\"')

    # remove invalid commas
    elif (tokid == token.OP) and ((tokval == '}') or (tokval == ']')):
      if (len(result) > 0) and (result[-1][1] == ','):
        result.pop()

    # detect single-line comments
    elif tokval == "//":
      sline_comment = True
      continue

    # detect multiline comments
    elif (last_token == '/') and (tokval == '*'):
      result.pop() # remove previous token
      mline_comment = True
      continue

    result.append((tokid, tokval))
    last_token = tokval

  return tokenize.untokenize(result)

invalid_escape = re.compile(r'\\[0-7]{1,6}')  # up to 6 digits for codepoints up to FFFF
def get_html(url, header):
    return BeautifulSoup(urllib2.urlopen(urllib2.Request(url,headers=header)), "html.parser")


def replace_with_codepoint(match):
    return unichr(int(match.group(0)[1:], 8))


def repair(brokenjson):
    return invalid_escape.sub(replace_with_codepoint, brokenjson)

class_arr = sys.argv
print class_arr
class_arr.pop(0)
for category in class_arr:
    query = category
    query = query.split()
    query = '+'.join(query)
    google_image_url = "https://www.google.com/search?q=" + query +  "&source=lnms&tbm=isch"
    bing_image_url = "https://www.bing.com/images/search?q=" + query
    print google_image_url
    print bing_image_url
    header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"
    }
    DIR = "image_sets"
    html_page = get_html(google_image_url, header)
    bing_html_page = get_html(bing_image_url, header)
    image_array = []
    combined_image_links = []
    for link in html_page.find_all("div", {"class": "rg_meta"}):
        image_link = json.loads(link.text)["ou"]
        image_page = json.loads(link.text)["ru"]
        combined_image_links.append(image_page)
        image_type = json.loads(link.text)['ity']
        image_array.append((image_link,image_type))
    for link in bing_html_page.find_all("a"):
        try:
            if link.has_attr('ihk'):
                image_obj= fixLazyJsonWithComments(link['m'])
                print image_obj
                image_link = json.loads(repair(image_obj))['imgurl']
                image_page = json.loads(repair(image_obj))['surl']
                combined_image_links.append(image_page)
                image_type = json.loads(repair(image_obj))['fmt']
                if image_link not in image_array:
                    image_array.append((image_link,image_type))
        except Exception as e:
            print "could not load link "
            print e
    if not os.path.exists(DIR):
        os.mkdir(DIR)
    DIR = os.path.join(DIR, query.split()[0])
    if not os.path.exists(DIR):
        os.mkdir(DIR)
    #Writing the urls of initial scrape into the output file.
    with open(query + "_images_link.json", "w") as outfile:
        json.dump({'output_url': combined_image_links}, outfile, indent=4)
    scraped_urls = []
    for index,img in enumerate(image_array):
        try:
            scraped_urls.append(img[0])
            req = urllib2.Request(img[0],headers={'User-Agent': header})
            raw_img = urllib2.urlopen(req).read()
            counter = len([i for i in os.listdir(DIR)]) + 1
            if len(img[1]) == 0:
                f = open(os.path.join(DIR, "image" + "_" + str(counter) + '.jpg'), 'wb')
            else:
                if img[1] == 'ashx':
                    f = open(os.path.join(DIR, "image" + "_" + str(counter) + '.jpg'), 'wb')
                else:
                    f = open(os.path.join(DIR, "image" + "_" + str(counter) + '.' + img[1]), 'wb')
            f.write(raw_img)
            f.close()
        except Exception as e:
            print "could not load : "+img[0]
            print e
    with open(query + "_scraped_images.json", "w") as outfile:
        json.dump({'scraped_urls': scraped_urls}, outfile, indent=4)
