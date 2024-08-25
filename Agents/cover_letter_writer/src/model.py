import asyncio
from functools import wraps
import os

import docx
from docx import Document
from langchain_core.prompts import PromptTemplate

from src.prompts import SKILLS_PROMPT, INTERESTS_PROMPT, CONSOLIDATOR_PROMPT
from src.config import model,resume,your_name


def async_invalid_output_handling(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            valid = False
            num_runs = 0
            while not valid and num_runs < 2:
                try:
                    await func(*args, **kwargs)
                    valid = True
                except ValueError as error:
                    print("An exception has occurred:",error," Running again.")
                    num_runs += 1
        return wrapper

def save_doc(doc_str:str,file_dir:str) -> None:
    doc_str = doc_str.replace("[Your Name]",your_name)
    doc = Document()
    font = doc.styles["Normal"].font
    font.name = "Calibri"
    font.size = docx.shared.Pt(12)
    para_list = doc_str.split("\n\n")
    for para in para_list:
        doc.add_paragraph(para)
    doc.save(file_dir)

class CoverLetterGenerator:
    def __init__(self) -> None:
        pass

    def load_job_info(self,selected=None) -> list:
        '''
        selected (list|None): list of jd file names, including file extensions, to load. None value -> all files
        '''
        jd_dict = {}
        if selected:
            for jd in selected:
                company_name = jd.split('.')[0]
                with open(os.path.join('job_descriptions',jd),'r',encoding='utf8') as f:
                    jd_text = f.read()
                with open(os.path.join('company_descriptions',jd),'r',encoding='utf8') as f:
                    company_text = f.read()
                jd_dict[company_name] = [jd_text,company_text]
        else:
            for jd in os.listdir('job_descriptions'):
                company_name = jd.split('.')[0]
                with open(os.path.join('job_descriptions',jd),'r',encoding='utf8') as f:
                    jd_text = f.read()
                with open(os.path.join('company_descriptions',jd),'r',encoding='utf8') as f:
                    company_text = f.read()
                jd_dict[company_name] = [jd_text,company_text]
        self.jd_dict = jd_dict

    async def _write_skills(self, extra_guidance:str,jd_val:str) -> str:
        prompt = PromptTemplate(
            template=SKILLS_PROMPT,
            input_variables=["extra_guidance","resume_val","jd_val"]
        )
        chain = prompt | model
        reply = await chain.ainvoke({"extra_guidance":extra_guidance,
                                  "resume_val":resume,
                                  "jd_val":jd_val})
        return reply.content
    
    async def _write_interests(self, extra_guidance:str,firm_val:str) -> str:
        prompt = PromptTemplate(
            template=INTERESTS_PROMPT,
            input_variables=["extra_guidance","resume_val","firm_val"]
        )
        chain = prompt | model
        reply = await chain.ainvoke({"extra_guidance":extra_guidance,
                                  "resume_val":resume,
                                  "firm_val":firm_val})
        return reply.content
    
    @async_invalid_output_handling
    async def generate_coverletter(self,extra_guidance:str=None) -> None:
        target_dir = "cover_letters"
        if os.path.isdir(target_dir) == False:
            os.makedirs(target_dir)

        for company in self.jd_dict:
            # Version 2
            print("Generating individual parts of cover letter")
            coros = [self._write_skills(extra_guidance=extra_guidance,jd_val=self.jd_dict[company][0]),\
                     self._write_interests(extra_guidance=extra_guidance,firm_val=self.jd_dict[company][1])]
            
            parts = await asyncio.gather(*coros)
            print("Compiling cover letter")
            consolidation_prompt = PromptTemplate(
                template=CONSOLIDATOR_PROMPT,
                input_variables=["skills_alignment","interests_alignment","jd_val"]
            )
            consolidation_chain = consolidation_prompt | model
            reply:str = consolidation_chain.invoke({
                "skills_alignment":parts[0],
                "interests_alignment":parts[1],
                "jd_val":self.jd_dict[company][0]
            }).content

            save_doc(reply,os.path.join(target_dir,"cover_letter_" + company + ".docx"))
    
    def generate_answer(self,prompt_file:str,jd:str) -> None:
        '''
        prompt_file (str): file path of required prompt
        jd (str): jd file of job concerned, e.g. tiktok.txt
        '''
        target_dir = "special_questions"
        if os.path.isdir(target_dir) == False:
            os.makedirs(target_dir)
        with open(os.path.join('job_descriptions',jd),'r',encoding='utf8') as f:
            jd_text = f.read()
        with open(os.path.join('special_prompts',prompt_file),'r',encoding='utf8') as f:
            prompt_str = f.read()
        prompt_str +=  '''
        ##Resume:
        {resume_val}
        
        ##Job Description:
        {jd_val}
        '''
        prompt = PromptTemplate.from_template(prompt_str)
        prompt_value = prompt.invoke({"resume_val":resume,"jd_val":jd_text})
        reply = model.invoke(prompt_value)
        reply_str = reply.content
        target_file_path = os.path.join(target_dir,"special_question_" + jd)
        f = open(target_file_path,"w")
        f.write(reply_str)
        f.close()