import argparse
import asyncio

from src.model import CoverLetterGenerator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",default="cover_letter",help="purpose of task - to write cover letter or otherwise. options: cover_letter | special_question")
    parser.add_argument("--selected_jobs",default=None,help="comma-delimited list of JD file names, e.g. tiktok.txt,temus.txt")
    parser.add_argument("--special_args",default=None,help="comma-delimited arguments if mode = special_question. Format: <prompt_file>,<jd_file>. E.g. question_tiktok.txt,tiktok.txt")
    parser.add_argument("--guidance",default=None,help="(Optional) Additional guidance to provide to the cover letter, if any.")
    opt = parser.parse_args()
    CL = CoverLetterGenerator()
    if opt.mode == "cover_letter":
        if opt.selected_jobs == None:
            selected = None
        else:
            selected = [item for item in opt.selected_jobs.split(",")]
        CL.load_job_info(selected)
        asyncio.run(CL.generate_coverletter(extra_guidance=opt.guidance))
    elif opt.mode == "special_question":
        question_args = opt.special_args.split(",")
        CL.generate_answer(prompt_file=question_args[0],jd=question_args[1])