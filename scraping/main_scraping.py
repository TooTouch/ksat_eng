from bs4 import BeautifulSoup
from tqdm import tqdm
import json
import os 
import argparse

from utils import setting, get_max_page_num, apply_test, clean_title

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EBS i 수능기출 크롤링 made by tootouch')
    parser.add_argument('--problem_cat',type=str,choices=['순서'],help='문제 유형')
    parser.add_argument('--ebs_id',type=str,help='EBS i ID')
    parser.add_argument('--ebs_passwd',type=str,help='EBS i PASSWORD')
    parser.add_argument('--save_dir',type=str,default='../data',help='save directory')
    parser.add_argument('--resume',action='store_true',help='resume')
    args = parser.parse_args() 

    # Define data structure & # resume
    if args.resume:        
        problem_collection = json.load(open(os.path.join(args.save_dir,'order.json'),'r'))    
    else:
        problem_collection = {args.problem_cat:[]}

    # setting
    driver = setting(args)

    # loop pages
    max_page_num = get_max_page_num(driver)
    for i in tqdm(range(max_page_num), desc='PAGE'):
        # i th page parsing
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        # application list
        apply_lst = soup.select_one('#div_contentList > table > tbody').find_all('tr')[1:]
        for apply in tqdm(apply_lst, desc='TEST', leave=False):
            # title, year, month, and category
            title = apply.find('td',attrs={'class':'t_info'}).find('strong').text.strip()
            year_month_category = clean_title(title=title)

            # resume
            if args.resume:
                if (year_month_category['year'] == problem_collection[args.problem_cat][-1]['year']) and \
                    (year_month_category['category'] == problem_collection[args.problem_cat][-1]['category']):
                    args.resume = False
                    continue
                else:
                    continue
            
            # scrapping problems
            apply_test(driver=driver, 
                       apply=apply, 
                       problem_category=args.problem_cat, 
                       problem_collection=problem_collection, 
                       **year_month_category)
            
        # next page
        if i+1 != max_page_num:
            driver.find_element_by_css_selector('#div_contentList > ul > div > a:nth-child(4)').click()
            driver.implicitly_wait(5)

        with open(os.path.join(args.save_dir,f'{args.problem_cat}.json'),'w') as f:
            json.dump(problem_collection, f, indent=4)