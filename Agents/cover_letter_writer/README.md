## Instructions to Run
### Preliminary steps:
1. For each company:
    a. Add a job description file to the job_descriptions folder.
    b. Add a company description file (including a brief description of the company and the key responsibilities in the job)
    c. Name each file as {company}.txt
2. Change directory to this folder and activate llm-env environment

### To write cover letter:
1. Run main.py. Specify jd files if only specific roles are concerned. 
E.g. python main.py --selected_jobs tiktok.txt,sap.txt
2. If required, specify additional guidance using the --guidance flag.
E.g. python main.py --selected_jobs tiktok.txt --guidance "Focus on ..."

### To answer a specific question:
1. Write prompt for question in .txt file. Save the file in the special_prompts folder.
2. Run main.py, setting mode as "special_question" and setting arguments for {prompt file},{jd file} in comma-delimited form. 
E.g. python main.py --mode special_question --special_args tiktok
_question.txt,tiktok.txt