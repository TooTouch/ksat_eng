from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select 
from bs4 import BeautifulSoup
from time import sleep

def clean_order_answers(answers: list):
    """
    Cleaning answers about 순서 문제

    Argument
    ---------
    answers: list
        webdriver elements about problem answers. ex)  [['① (A) - (B) - (C)'], ... ,[...]]

    Return
    ------
    result_lst: list
        cleaned answers. ex) [['A','B','C'], ... , [...]]
    """
    remove_lst = ['①','②','③','④','⑤','(',')']
    result_lst = []

    for answer in answers:
        answer_text = answer.text

        for remove_w in remove_lst:
            answer_text = answer_text.replace(remove_w,'')
        
        if '-' in answer_text:
            answer_text = [a.strip() for a in answer_text.split('-')]
        elif '－' in answer_text:
            answer_text = [a.strip() for a in answer_text.split('－')]
        result_lst.append(answer_text)
    
    return result_lst

def clean_order_context(context_lst: list, remove_number: bool = False):
    """
    Cleaning context about 순서 문제

    Arguments
    ---------
    context_lst: list
        webdriver elements about context.
    remove_number: bool
        optional, if True removes problem number in context. ex) 37. we ~ 

    Returns
    ------
    context_first: str
        first sentence for 순서 문제 
    context_sub: list
        sentences for 순서 문제. ex) {'A': ... ,'B': ... ,'C': ...}
    """
    context_sub = dict()

    context = " ".join([c.text if '*' not in c.text else c.text[:c.text.find('*')] for c in context_lst])
    # remove uninformative characters 
    context = context.replace('\xad','')

    # first
    context_first = context.split('(A)')[0].strip()
    if remove_number: # 묶음 문제의 경우 and 지문 안에 번호가 섞여 있는 경우
        context_first = context_first[context_first.find(str(remove_number))+3:].strip()
    # A
    context_sub['A'] = context.split('(A)')[1].split('(B)')[0].strip()
    # B
    context_sub['B'] = context.split('(A)')[1].split('(B)')[1].split('(C)')[0].strip()
    # C
    context_sub['C'] = context.split('(A)')[1].split('(B)')[1].split('(C)')[1].strip()

    return context_first, context_sub

def clean_answer_info(answer_info: list):
    """
    Cleaning answer information

    Argument
    --------
    answer_info: list
        webdriver elements about answer infomation 

    Returns
    -------
    correct_answer: int
        correct answer number, not index. ex) 5
    correct_rate_problem: float
        correct percentage about problem. ex) 53.2
    select_rate_answers: list
        select rate of each answer. ex) [21.2, 41.3, 10.5, 27.0]

    """
    answer_numbers = '①②③④⑤'

    correct_answer = int(answer_numbers.find(answer_info[1].find_all('td')[0].text)) + 1
    correct_rate_problem = float(answer_info[1].find_all('td')[1].text.replace('%',''))
    select_rate_answers = [float(s.text.replace('%','')) for s in answer_info[2].find_all('td')]

    return correct_answer, correct_rate_problem, select_rate_answers

def clean_title(title: str):
    """
    Extract information from title

    Argument
    --------
    title: str
        application title. ex) 2020년 고3 10월 학평(서울) 영어

    Return
    ------
    title_info: dict
        title information. ex) {'year':2020, 'month':10, 'category':'학평-서울'}

    """
    title = title.split()
    year = None
    month = None
    category = None

    for t in title:
        if '년' in t:
            year = int(t.replace('년','').strip())

        if '월' in t:
            month = int(t.replace('월','').strip())

        if '수학능력시험' in t:
            category = '수능'
        elif ('모평' in t) or ('모의평가' in t):
            category = '모평'
        elif ('학평' in t) or ('학력평가' in t):
            category = '학평'
        
        if '홀수' in t:
            category += '-홀수'
        elif '짝수' in t:
            category += '-짝수'

        if 'A형' in t:
            category += '-A형'
        elif 'B형' in t:
            category += '-B형'

        if '서울' in t:
            category += '-서울'
        elif '경기' in t:
            category += '-경기'
        elif '인천' in t:
            category += '-인천'

    title_info = {'year':year, 'month':month, 'category':category}

    return title_info

def scraping_text(problem, problem_category: str, problem_collection: dict, **kwargs):
    """
    Scraping information in the test website

    Arguments
    ---------
    problem
        webdriver element about problem
    problem_category: str
        selected category to extract information in the test. ex) '순서'
    problem_collection: dict
        dictionary to append problem information
    
    Return
    ------
    question: str or None
        if problem is '묶임 문제' returns question, otherwise return None
    question_numbers: list or None
        if problem is '묶임 문제' returns question number list, otherwise return None
    """
    # Define data structure
    problem_dict = {
        'question': str,
        'context': str,
        'context_sub': dict(),
        'answers': list(),
        'correct_answer': int,
        'correct_rate_problem': float,
        'select_rate_answers': list(),
        'problem_number': int,
        'problem_score': int,
        'year': int,
        'month': int,
        'category': str # [수능-짝수, 수능-홀수, 모평, 학평(경기), 학평(서울)]
    }

    # find question and paragraphs
    problem_i = problem.find(attrs={'name':'divQuestion'})

    # 문제 여러개를 묶는 경우
    # ex) [36 ~ 37]  주어진 글 다음에 이어질 글의 순서로 가장 적절한 것을 고르시오.
    if problem_i is None:
        question = problem.find(attrs={'name':'divSentence'}).text
        if question == '': # 0번째 인덱스로 divSentence가 들어가서 아무것도 없는 경우
            return None, []
        elif (question.find('[') == -1) or (question.find('[') != 0): # 묶임 문제지만 번호가 안적혀 있는 경우
            return None, []
        else:
            # ex) [36 ~ 37] -> [36, 37]
            question_numbers = [int(n.strip()) for n in question.split(']')[0].replace('[','').split('~')]

            # ex) 주어진 글 다음에 이어질 글의 순서로 가장 적절한 것을 고르시오.
            question = question.split(']')[1].strip()

            return question, question_numbers

    else:
        # problem_lst = problem_i.find_all('div', attrs={'name':'divParagraph'}, recursive=False)
        problem_lst = []
        for child in problem_i.children:
            if child.name == 'ul': # ul is formed by answer elements
                break
            problem_lst.append(child)
        
        # question & problem_number
        if kwargs['question'] is not None: # 묶임 문제인 경우
            question_number = kwargs['question_numbers'].pop(0)
            question = kwargs['question']
        else:
            if problem_lst[0].text == '': # 첫 div가 비어있는 경우
                del problem_lst[0]
            question_number = int(problem_lst[0].text.split('.')[0].strip())
            question = problem_lst[0].text.split('.')[1].strip()
                
        # long sentence start number
        max_question_number = 46 if kwargs['year'] < 2013 else 41 
        # listening end number
        min_question_number = 22 if kwargs['year'] == 2013 else 17

        if (problem_category in question) and \
           (question_number < max_question_number) and \
           (question_number > min_question_number):

            # question & problem_number
            problem_dict['problem_number'] = question_number
            problem_dict['question'] = question

            # context
            if kwargs['question'] is not None:
                if str(question_number) == problem_lst[0].text.replace('.','').strip():
                    context_first, context_sub = clean_order_context(context_lst=problem_lst[1:])
                else:
                    context_first, context_sub = clean_order_context(context_lst=problem_lst, remove_number=question_number)
            else:
                # 첫 지문이 이미지인 경우
                if problem_lst[1].find('img') is not None:
                    if len(kwargs['question_numbers']) == 0:
                        return None, []
                    else:
                        return kwargs['question'], kwargs['question_numbers']

                context_first, context_sub = clean_order_context(context_lst=problem_lst[1:])
            problem_dict['context'] = context_first
            problem_dict['context_sub'] = context_sub

            # answers 
            answer_lst = problem_i.find('ul').find_all('li')
            problem_dict['answers'] = clean_order_answers(answers=answer_lst)

            # answer_info
            answer_info = problem.select_one('table.boardcorrect > tbody') .find_all('tr')
            correct_answer, correct_rate_problem, select_rate_answers = clean_answer_info(answer_info=answer_info)
            problem_dict['correct_answer'] = correct_answer
            problem_dict['correct_rate_problem'] = correct_rate_problem 
            problem_dict['select_rate_answers'] = select_rate_answers

            # problem_score
            problem_score = problem_i.find('ul').find('div')
            problem_dict['problem_score'] = int(problem_score.text.split('점')[0].replace('[',''))

            # year, month, and category
            problem_dict['year'] = kwargs['year']
            problem_dict['month'] = kwargs['month']
            problem_dict['category'] = kwargs['category']

            # append problem_dict
            problem_collection[problem_category].append(problem_dict)

            if len(kwargs['question_numbers']) == 0:
                return None, []
            else:
                return kwargs['question'], kwargs['question_numbers']

        return None, []


def apply_test(driver, apply, problem_category: str, problem_collection: dict, **kwargs):
    """
    Apply the test in EBS i

    Arguments
    ---------
    driver
        webdriver
    apply
        webdriver element for applying the test
    problem_category: str
        selected category to extract information in the test. ex) '순서'
    problem_collection: dict
        dictionary to append problem information
    
    """
    # apply test 
    apply_src = apply.find('a',attrs={'class':'btn_apply'})['href']
    driver.execute_script(apply_src)
    driver.implicitly_wait(10)
    sleep(3)

    # switch window to new window
    driver.switch_to.window(driver.window_handles[-1])

    # change view mode
    driver.find_element_by_id('mode2').click()
    driver.implicitly_wait(10)

    # parsing page_source using bs4
    req = driver.page_source
    soup = BeautifulSoup(req, 'html.parser')
    problems = soup.select_one('div.study-mode').select('div.question-container')
    
    # define 묶임 문제 default 
    kwargs['question'] = None
    kwargs['question_numbers'] = []

    for idx, problem in enumerate(problems):
        # 묶음 문제인데 problem_category에 해당하지 않는 경우 pass
        if (kwargs['question'] is not None) and (problem_category not in kwargs['question']):
            if len(kwargs['question_numbers']) > 0:
                del kwargs['question_numbers'][0]
                continue
            else:
                kwargs['question'] = None
                kwargs['question_numbers'] = []

        question, question_numbers = scraping_text(problem=problem, problem_category=problem_category, problem_collection=problem_collection, **kwargs)

        # update 묶음 문제    
        kwargs['question'] = question
        kwargs['question_numbers'] = question_numbers

        
    # come back root
    driver.switch_to.window(driver.window_handles[0])


def setting(args):
    """
    Environment setting for scraping EBS i

    Argument
    --------
    args
        parsered arguments

    Return
    ------
    driver
        webdriver 
    """
    # webdriver setting
    chrome_options = Options()
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--headless")

    driver = webdriver.Chrome(options=chrome_options)

    # EBSi Login page
    driver.implicitly_wait(3)
    driver.get('http://www.ebsi.co.kr/ebs/pot/potl/login.ebs?destination=/index.jsp&alertYn=N')

    ## Enter user id and passwd 
    driver.find_element_by_name('userid').send_keys(args.ebs_id)
    driver.find_element_by_name('passwd').send_keys(args.ebs_passwd)

    ## click login button
    driver.find_element_by_class_name('loginWrap').find_element_by_tag_name('button').click()

    # 기출문제 page
    driver.implicitly_wait(3)
    driver.get('http://www.ebsi.co.kr/ebs/xip/xipc/previousPaperList.ebs')

    # select 전체 from 연도선택
    select = Select(driver.find_element_by_id('yearNum'))
    select.select_by_value("")

    # select 영어
    driver.find_element_by_id('subTab_eng').click()

    return driver


def get_max_page_num(driver):
    """
    Obtain maximum page number

    Argument
    --------
    driver
        webdriver logined and accessed in EBS i
    
    Return
    ------
    max_page_num: int
        maximum page number
    
    """
    last_page_nav = driver.find_element_by_xpath('//*[@id="div_contentList"]/ul/div/a[2]').get_attribute('onclick')
    max_page_num = last_page_nav.split('(')[1]
    max_page_num = int(max_page_num[:max_page_num.find(')')])

    return max_page_num

    
