---
title: Smart Resume Analyzer
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Scores a resume against a job description and explains why.
---

# Smart Resume Analyzer

Upload a resume (PDF or DOCX) and paste a job description. The app scores the match,
breaks the score down into skills, requirement coverage, experience relevance and
education, checks whether an applicant tracking system can read the resume, and can
suggest a learning path and stronger wordings.

Source code, documentation and the evaluation behind the scores:
[github.com/kanishk2705/Smart-Resume-Analyzer](https://github.com/kanishk2705/Smart-Resume-Analyzer)

**Privacy in this demo:** your resume file is not stored. The analysis result is kept on
this Space's temporary disk so you can reopen it during your visit; other visitors cannot
see it, but anyone you give its share link to can. Using the AI coach sends the job title,
missing skill names and flagged resume statements to Groq.
