"""Realistic synthetic CV documents for simulated candidates.

All names, companies and universities are invented. A CV states skill levels the way real
CVs do: a level word ("Python (senior)", "advanced Docker"), years of experience
("SQL: 5 years", "5+ years of Python"), or nothing at all ("Git"). Exaggerating candidates
inflate titles and years too.
"""
from . import LEVELS, skill_bank

FIRST_NAMES = ["Amira", "Omar", "Lina", "Youssef", "Sara", "Karim", "Nour", "Hana", "Ali", "Maya",
               "Daniel", "Priya", "Chen", "Sofia", "Lucas", "Fatima", "Mateo", "Aisha", "Jonas", "Elif"]
LAST_NAMES = ["Hassan", "Farouk", "Nasser", "Haddad", "Mansour", "Kamal", "Rahman", "Okafor", "Novak",
              "Silva", "Kim", "Patel", "Garcia", "Muller", "Yilmaz", "Ibrahim", "Lopez", "Tanaka"]
CITIES = ["Cairo", "Alexandria", "Dubai", "Berlin", "Lisbon", "Toronto", "Remote"]
COMPANIES = ["Nilewave Payments", "Deltasoft", "Bluegate Logistics", "Orbitly", "Sandstone Health",
             "Quartzline", "Papyrus Labs", "Harbor & Pine", "Lumen Retail", "Cedar Analytics",
             "Brightpath Learning", "Northwind Mobility"]
UNIVERSITIES = ["Nile Valley University", "Delta Institute of Technology", "Coastal Polytechnic",
                "Riverside University", "Eastern College of Engineering"]
DEGREES = ["B.Sc. Computer Science", "B.Sc. Computer Engineering", "B.Sc. Information Systems",
           "M.Sc. Software Engineering", "B.Sc. Mathematics"]
# Other tools that make a CV look real; none of them is a skill the pack checks.
FILLER_TOOLS = ["Linux", "Jira", "Agile/Scrum", "AWS", "CI/CD pipelines", "Figma", "Terraform", "Bash", "Redis"]

ROLE_BY_SKILL = {
    "python": "Backend Engineer", "sql": "Data Engineer", "javascript": "Frontend Developer",
    "react": "Frontend Developer", "docker": "DevOps Engineer", "git": "Software Engineer",
    "machine-learning": "ML Engineer", "rest-apis": "Backend Engineer",
}
TITLE_PREFIX = {1: "Trainee", 2: "Junior", 3: "", 4: "Senior"}
YEARS_FOR_LEVEL = {1: (0, 1), 2: (1, 2), 3: (3, 4), 4: (5, 9)}
LEVEL_WORDS_OUT = {1: ["beginner", "basic"], 2: ["junior"], 3: ["intermediate", "mid-level"],
                   4: ["senior", "advanced", "expert"]}

BULLETS = {
    "python": ["Built internal services in {alias} handling {n}k requests a day",
               "Wrote data-cleaning scripts in Python that cut report time by {n}%"],
    "sql": ["Designed {alias} schemas and tuned slow queries ({n}x faster)",
            "Maintained reporting views over {n} million rows"],
    "javascript": ["Rebuilt the checkout flow in {alias}, lifting conversion by {n}%",
                   "Wrote JavaScript widgets used on {n} partner sites"],
    "react": ["Migrated {n} pages to {alias} components with shared state",
              "Built a React design system used by {n} teams"],
    "docker": ["Containerised {n} services with Docker and docker-compose",
               "Cut image sizes by {n}% with multi-stage Docker builds"],
    "git": ["Introduced a Git branching model for a team of {n}",
            "Set up code review and merge rules in {alias} for {n} repositories"],
    "machine-learning": ["Trained churn models with scikit-learn (AUC +{n} points)",
                         "Shipped a machine learning ranking model to {n}k users"],
    "rest-apis": ["Designed REST APIs consumed by {n} mobile and web clients",
                  "Versioned and documented {n} public REST endpoints"],
}
ALIAS_IN_BULLET = {"python": ["Python", "Django", "FastAPI"], "sql": ["PostgreSQL", "MySQL"],
                   "javascript": ["JavaScript", "TypeScript"], "react": ["React", "Next.js"],
                   "git": ["GitHub", "GitLab"]}


def skill_phrase(rng, name, level):
    """How the skills section states one skill. Returns (text, stated level or None)."""
    style = rng.choices(["word", "years", "years-before", "bare"], weights=[4, 3, 2, 2])[0]
    if style == "bare" or level == 0:
        return name, None
    if style in ("years", "years-before") and level >= 2:
        low, high = YEARS_FOR_LEVEL[level]
        years = rng.randint(max(low, 1), high)
        unit = "year" if years == 1 else "years"
        if style == "years":
            return rng.choice([f"{name}: {years} {unit}", f"{name} ({years} {unit})"]), level
        plus = "+" if level == 4 else ""
        return f"{years}{plus} {unit} of {name}", level
    word = rng.choice(LEVEL_WORDS_OUT[level])
    return rng.choice([f"{name} ({word})", f"{word} {name}", f"{name} - {word}"]), level


def make_cv(rng, persona):
    """Write the CV text for ``persona`` (uses ``claimed`` levels) and return
    (text, stated) where ``stated`` maps skill -> the level the CV states (None if not stated)."""
    claimed = persona["claimed"]
    skills = list(claimed)
    top_skill = max(skills, key=lambda k: claimed[k])
    seniority = max(1, claimed[top_skill])
    role = ROLE_BY_SKILL[top_skill]
    first, last = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
    title = (TITLE_PREFIX[seniority] + " " + role).strip()
    total_years = rng.randint(*YEARS_FOR_LEVEL[seniority]) if seniority > 1 else 0

    lines = [f"{first} {last}", f"{title} · {rng.choice(CITIES)} · {first.lower()}.{last.lower()}@example.com", ""]
    lines.append("SUMMARY")
    experience = f"{total_years}+ years of experience" if total_years else "recent graduate"
    lines.append(f"{role} ({experience}) who enjoys building reliable products and working with "
                 f"{rng.choice(['small teams', 'product managers', 'designers', 'data teams'])}.")
    lines.append("")

    lines.append("EXPERIENCE")
    year, jobs = 2026, max(1, min(3, total_years // 2 + 1))
    companies = rng.sample(COMPANIES, jobs)
    lengths = [total_years // jobs + (1 if j < total_years % jobs else 0) for j in range(jobs)]
    for j, company in enumerate(companies):
        start = year - max(lengths[j], 1)
        job_title = title if j == 0 else (TITLE_PREFIX[max(1, seniority - 1)] + " " + role).strip()
        end = "present" if j == 0 else str(year)
        lines.append(f"{job_title} - {company} ({start} - {end})")
        for skill in rng.sample(skills, min(2, len(skills))):
            template = rng.choice(BULLETS[skill])
            alias = rng.choice(ALIAS_IN_BULLET.get(skill, [skill_bank()[skill]["name"]]))
            lines.append("- " + template.format(alias=alias, n=rng.randint(2, 40)))
        lines.append(f"- Worked with {rng.choice(FILLER_TOOLS)} and {rng.choice(FILLER_TOOLS)}")
        year = start
    lines.append("")

    lines.append("EDUCATION")
    lines.append(f"{rng.choice(DEGREES)} - {rng.choice(UNIVERSITIES)} ({year - rng.randint(0, 2)})")
    lines.append("")

    lines.append("SKILLS")
    stated, parts = {}, []
    for skill in skills:
        text, level = skill_phrase(rng, skill_bank()[skill]["name"], claimed[skill])
        parts.append(text)
        stated[skill] = level
    parts += rng.sample(FILLER_TOOLS, 2)
    rng.shuffle(parts)
    lines.append(", ".join(parts))
    return "\n".join(lines), stated


def make_job(rng, skills):
    """A short job posting asking for some of the candidate's skills and one other."""
    others = [k for k in skill_bank() if k not in skills]
    wanted = rng.sample(skills, min(2, len(skills))) + rng.sample(others, 1)
    names = [skill_bank()[k]["name"] for k in wanted]
    company = rng.choice(COMPANIES)
    return (f"{company} is hiring. You will build and run our core product. "
            f"Must have: {names[0]} and {names[1] if len(names) > 1 else names[0]}. Nice to have: {names[-1]}.")


def level_name(level):
    return LEVELS[level]
