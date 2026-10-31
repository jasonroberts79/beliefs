A = ("owner:me", "repo:web")
B = ("owner:me", "repo:engine")
C = ("owner:me", "repo:notes")

SCENARIOS = [
    (
        "near-duplicate ports are different facts",
        [
            ("The web app's Vite dev server listens on port 5173.", "tool_observed", "s1", A),
            ("The web app's backend API server listens on port 8080.", "tool_observed", "s1", A),
        ],
        [
            (A, "Which port does the Vite dev server use?", "5173", "has"),
            (A, "Which port does the backend API server use?", "8080", "has"),
        ],
    ),
    (
        "negation is a contradiction, not a paraphrase",
        [
            ("The engine's integration tests require a running Redis server.", "tool_observed", "s1", B),
            ("The engine's integration tests do not require a running Redis server.", "tool_observed", "s2", B),
        ],
        [
            (B, "Do the engine's integration tests need Redis?", "Redis", "loose"),
        ],
    ),
    (
        "repo exception does not leak to sibling repos",
        [
            ("I want all my Python code formatted with Black.", "user_said", "s1", A),
            ("In this repo, format Python code with Ruff.", "user_said", "s2", B),
            ("Remember, this repo uses Ruff as its formatter.", "user_said", "s3", B),
        ],
        [
            (B, "Which formatter should Python code use?", "Ruff", "has"),
            (C, "Which formatter should Python code use?", "Black", "has"),
        ],
    ),
    (
        "mentioning an alternative is not a switch",
        [
            ("I use Neovim for all my editing.", "user_said", "s1", A),
            ("My editor is Neovim.", "user_said", "s2", B),
            ("A few teammates use VS Code, so keep the .vscode folder checked in.", "user_said", "s3", A),
        ],
        [
            (A, "Which editor does the user use?", "Neovim", "firm"),
        ],
    ),
    (
        "retracting the source of a supersede restores the old fact",
        [
            ("This repo deploys to Heroku.", "tool_observed", "s1", A),
            ("We moved this repo's deployments from Heroku to Fly.io.", "user_said", "s2", A),
            ("RETRACT", "s2"),
        ],
        [
            (A, "Where does this repo deploy?", "Fly", "gone"),
            (A, "Where does this repo deploy?", "Heroku", "has"),
        ],
    ),
    (
        "transient chatter leaves no belief",
        [
            ("Brb, grabbing a coffee while this builds.", "user_said", "s1", A),
            ("I'm heading to lunch in five minutes, but remember that I always want commit messages in Conventional Commits format.", "user_said", "s1", A),
        ],
        [
            (A, "Is the user getting coffee?", "coffee", "gone"),
            (A, "When is the user going to lunch?", "lunch", "gone"),
            (A, "What commit message format does the user want?", "Conventional", "has"),
        ],
    ),
    (
        "repeated agent inferences never make a belief firm",
        [
            ("The user seems to prefer a functional programming style.", "agent_inferred", "s1", A),
            ("The user probably prefers functional style over classes.", "agent_inferred", "s2", B),
            ("The user appears to favor functional programming.", "agent_inferred", "s3", C),
            ("The user seems to prefer a functional programming style.", "agent_inferred", "s4", A),
        ],
        [
            (C, "What programming style does the user prefer?", "functional", "loose"),
        ],
    ),
    (
        "tool output beats a stale README",
        [
            ("Install dependencies with npm install.", "document", "web/README.md", A),
            ("This repo uses pnpm: it has a pnpm-lock.yaml and no package-lock.json.", "tool_observed", "s1", A),
        ],
        [
            (A, "Which package manager does this repo use?", "pnpm", "has_or_flagged"),
        ],
    ),
    (
        "supersede in one repo leaves the same fact in another repo",
        [
            ("This repo runs its tests with Jest.", "tool_observed", "s1", A),
            ("This repo runs its tests with Jest.", "tool_observed", "s2", B),
            ("We migrated this repo's tests from Jest to Vitest.", "user_said", "s3", A),
        ],
        [
            (A, "Which test runner does this repo use?", "Vitest", "has"),
            (B, "Which test runner does this repo use?", "Jest", "has"),
        ],
    ),
    (
        "retraction blocks future evidence from the same source",
        [
            ("The engine builds with CMake.", "document", "docs/old-setup.md", B),
            ("RETRACT", "docs/old-setup.md"),
            ("The engine's CI runs on Jenkins.", "document", "docs/old-setup.md", B),
        ],
        [
            (B, "What build system does the engine use?", "CMake", "gone"),
            (B, "Where does the engine's CI run?", "Jenkins", "gone"),
        ],
    ),
]
