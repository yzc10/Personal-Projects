SKILLS_PROMPT = """
## Task:
Write 1 long paragraph detailing how this candidate's skills and work experiences make him suitable for the following job. 

## Context:
This text will be weaved into the candidate's cover letter for the job.

## Important Requirements:
- Write sincerely with a personal voice. Write in first-person point of view. 
- Keep the expressions and sentence structures varied and flexible. 
- Be specific and include details. 
- Pay specific attention to his academic and professional background. 
- *** INCLUDE OTHER DETAILS AS APPROPRIATE***
- Include only the output text.

## Other Guidance (if any):
{extra_guidance}

## Resume:
{resume_val}

## Job Description:
{jd_val}

## Output:
"""

INTERESTS_PROMPT = """
## Task:
Write 1 long paragraph detailing how this candidate's interests, passions and values align with the company and the role. 

## Context:
This text will be weaved into the candidate's cover letter for the job.

## Important Requirements:
- Write sincerely with a personal voice. Write in first-person point of view. 
- Keep the expressions and sentence structures varied and flexible. 
- Focus on how the candidate's passions align with the role and/or the organization. 
- *** INCLUDE OTHER DETAILS AS APPROPRIATE***
- Include only the output text.

## Other Guidance (if any):
{extra_guidance}

## Resume:
{resume_val}

## Firm and Job Description:
{firm_val}

## Output:
"""

CONSOLIDATOR_PROMPT = """
## Task:
Combine the following paragraphs into one coherent cover letter for the following job:

## General Structure of Cover Letter:
Intro (1 paragraph) - Skill alignment section - Interests alignment section - Conclusion

## Important Requirements:
- Ensure that the whole cover letter flows. The paragraphs should read smoothly from one to the other.
- Write sincerely. Keep the expressions and sentence structures varied and flexible. 
- Include only the text, starting with Dear and ending off with Best Regards.

## Skill Alignment Section:
{skills_alignment}

## Interest Alignment Section:
{interests_alignment}

## Job Description:
{jd_val}

## Cover Letter:
"""