"""
Holds `Check`, the class every check `repo:audit` runs extends, and the
checks themselves, each one invariant the audit holds the checkout to and
each found through `Check.__subclasses__()` rather than through a list kept
beside them.
"""

from collections            import Counter
from collections.abc        import Iterator
from filecmp                import cmp
from functools              import cached_property
from packaging.requirements import Requirement
from packaging.utils        import canonicalize_name
from pathlib                import Path
from pydantic               import BaseModel, ValidationError
from re import findall, search, sub

from thermur.repo.checkout import Checkout
from thermur.repo.schemas  import Finding, Parity


class Check(BaseModel, extra="forbid", frozen=True, use_attribute_docstrings=True):
    """
    One invariant the audit holds the checkout to, read through `checkout`.
    """

    checkout: Checkout
    """
    The clone the check reads.
    """

    @property
    def findings(self) -> list[Finding]:
        """
        Collects every finding `scan` makes, reporting a document that fails
        to validate as one finding per field on the file the error's title
        names, and letting through an error whose title names no file.
        """
        try:
            return list(self.scan())
        except ValidationError as error:
            if not (self.checkout.root / error.title).is_file():
                raise

            return [
                Finding(
                    file    = Path(error.title),
                    message = (
                        f"`{'.'.join(map(str, detail['loc']))}` in "
                        if detail["loc"]
                        else ""
                    ) + f"`{error.title}`: {detail['msg']}"
                )
                for detail in error.errors()
            ]

    @property
    def pairs(self) -> list[Parity]:
        """
        Pairs each declaration a file restates with the one it restates,
        which a check comparing no restated declarations leaves empty.
        """
        return []

    def scan(self) -> Iterator[Finding]:
        """
        Reads the checkout for every place it breaks this invariant, which
        for a check comparing restated declarations is each pair whose copy
        reads anything but the original.
        """
        yield from filter(None, (pair.finding for pair in self.pairs))


class BinCheck(Check):
    """
    Every program in the folder the `_.path` of `.mise/config.toml` puts on
    the path is a regular file holding the same bytes as the first of them.

    Git checks a symlink out as a plain, non-executable file holding its
    target's name wherever `core.symlinks` is off, so a shell looking the
    program up on the path skips it and runs any program of that name the
    machine already holds.
    """

    def scan(self) -> Iterator[Finding]:
        """
        Names each program that is a symlink, then each copy whose bytes
        differ from the first copy's.
        """
        root   = self.checkout.root
        linked = [path for path in self.checkout.scripts if (root / path).is_symlink()]
        for path in linked:
            yield Finding(
                file    = path,
                message = f"`{path}` is a symlink rather than a copy"
            )

        copies = [path for path in self.checkout.scripts if path not in linked]
        for path in copies[1:]:
            if not cmp(root / copies[0], root / path, shallow=False):
                yield Finding(
                    file    = path,
                    message = f"`{path}` differs byte for byte from `{copies[0]}`"
                )


class LabelCheck(Check):
    """
    Every file naming a label agrees with `.github/labels.toml`, meaning the
    release-notes categories file each label it declares under one category,
    the issue templates and the contributor guide name only labels it
    declares, and the guide describes each label as the registry does.
    """

    @cached_property
    def described(self) -> dict[str, str]:
        """
        Maps each label the contributor guide's table names to the
        description beside it, the table running from its header to the
        first blank line.
        """
        table = self.checkout.read(self.guide).partition(
            "| **Label** | **Covers** |"
        )[2]
        return dict(findall(r"(?m)^\| `(.+?)` \| (.+) \|$", table.partition("\n\n")[0]))

    @property
    def guide(self) -> Path:
        """
        Names the contributor guide, relative to the root.
        """
        return Path(".github/CONTRIBUTING.md")

    @property
    def named(self) -> dict[Path, list[str]]:
        """
        Maps each file naming labels to every label it names, in the order
        it names them, meaning the release-notes categories, each issue
        template, and the contributor guide.
        """
        release = self.checkout.release
        return (
            {release.file: release.names}
            | {
                path: template.labels
                for path, template in self.checkout.templates.items()
            }
            | {self.guide: list(self.described)}
        )

    @property
    def pairs(self) -> list[Parity]:
        """
        Pairs the description the contributor guide gives each label with
        the one the registry declares.
        """
        labels = self.checkout.labels
        return [
            Parity(
                copied   = self.described.get(label.name),
                file     = self.guide,
                label    = f"The description of `{label.name}` in `{self.guide}`",
                origin   = f"its description in `{labels.file}`",
                original = label.description
            )
            for label in labels.rows
        ]

    def scan(self) -> Iterator[Finding]:
        """
        Names each label a file names that the registry does not declare,
        then each declared label the categories file under any number of
        categories but one, then each label the guide describes otherwise.
        """
        labels = self.checkout.labels
        for file, names in self.named.items():
            for name in dict.fromkeys(names):
                if name not in labels.names:
                    yield Finding(
                        file    = file,
                        message = f"`{file}` names `{name}`, a label `{labels.file}` "
                        "does not declare"
                    )

        release = self.checkout.release
        filed   = Counter(release.names)
        for name in labels.names:
            if filed[name] != 1:
                yield Finding(
                    file    = release.file,
                    message = f"`{release.file}` files `{name}` under "
                    f"{filed[name] or 'no'} release-notes categories rather than one"
                )

        yield from super().scan()


class ParityCheck(Check):
    """
    Every file restating the Python version, the uv release, the license,
    or the release `CITATION.cff` cites reads the value `pyproject.toml` or
    `.mise/config.toml` declares.

    The manifest's `requires-python` floor is the Python version the
    README's badge, the formatter's `target-version`, and the minor of
    the `python` mise pins each restate. The `uv` mise pins is the release
    `[tool.uv]` requires, and the manifest's license and authors are what
    the README's badge and the title and the copyright line of `LICENSE`
    restate. `CITATION.cff` restates the manifest's version, license,
    repository, and authors.
    """

    @property
    def pairs(self) -> list[Parity]:
        """
        Pairs each restated declaration with the one it restates.
        """
        citation     = self.checkout.citation
        config       = self.checkout.config
        manifest     = self.checkout.manifest
        license_path = Path("LICENSE")
        readme       = self.checkout.read(manifest.readme)
        notice       = self.checkout.read(license_path)
        badge        = self.extract(r"badge/License-((?:--|[^-])+)-", readme)
        python       = f"the floor of `requires-python` in `{manifest.file}`"

        return [
            Parity(
                copied   = self.extract(r"badge/python-(\d+\.\d+)\+", readme),
                file     = manifest.readme,
                label    = f"The Python badge in `{manifest.readme}`",
                origin   = python,
                original = manifest.floor
            ),
            Parity(
                copied   = manifest.target_version,
                file     = manifest.file,
                label    = "The `target-version` in `[tool.prose]`",
                origin   = python,
                original = manifest.floor
            ),
            Parity(
                copied   = config.python,
                file     = config.file,
                label    = f"The minor of the `python` pin in `{config.file}`",
                origin   = python,
                original = manifest.floor
            ),
            Parity(
                copied   = manifest.required_version,
                file     = manifest.file,
                label    = "The `required-version` in `[tool.uv]`",
                origin   = f"the `uv` pin in `{config.file}` as a `==` requirement",
                original = f"=={config.uv}"
            ),
            Parity(
                copied   = badge.replace("--", "-") if badge else None,
                file     = manifest.readme,
                label    = f"The License badge in `{manifest.readme}`",
                origin   = f"the `license` in `{manifest.file}`",
                original = manifest.license
            ),
            Parity(
                copied   = notice.partition("\n")[0],
                file     = license_path,
                label    = f"The title of `{license_path}`",
                origin   = f"the full name of the `license` in `{manifest.file}`",
                original = f"{manifest.license} License"
            ),
            Parity(
                copied   = self.extract(r"Copyright \(c\) [\d-]+ (.+)", notice),
                file     = license_path,
                label    = f"The copyright holder list in `{license_path}`",
                origin   = f"the author list in `{manifest.file}`",
                original = manifest.holders
            ),
            *(
                Parity(
                    copied   = copied,
                    file     = citation.file,
                    label    = f"The `{key}` in `{citation.file}`",
                    origin   = f"the `{name}` in `{manifest.file}`",
                    original = original
                )
                for key, copied, name, original in (
                    ("license", citation.license, "license", manifest.license),
                    (
                        "repository-code", citation.repository,
                        "Repository", manifest.repository
                    ),
                    ("version", citation.version, "version", manifest.version)
                )
            ),
            Parity(
                copied   = citation.holders or None,
                file     = citation.file,
                label    = f"The author list in `{citation.file}`",
                origin   = f"the author list in `{manifest.file}`",
                original = manifest.holders
            )
        ]

    @staticmethod
    def extract(pattern: str, text: str) -> str | None:
        """
        Reads the first group `pattern` captures in `text`, or returns
        `None` where nothing matches.
        """
        return match[1] if (match := search(pattern, text)) else None


class PinCheck(Check):
    """
    The hatchling build requirement and every build constraint `[tool.uv]`
    declares each name one version, since uv resolves hatchling and its own
    dependencies outside `uv.lock`.
    """

    @staticmethod
    def exact(requirement: Requirement) -> bool:
        """
        Tells whether `requirement` names one version, meaning a single `==`
        specifier carrying no `.*` wildcard.
        """
        return [
            (specifier.operator, specifier.version.endswith(".*"))
            for specifier in requirement.specifier
        ] == [("==", False)]

    def scan(self) -> Iterator[Finding]:
        """
        Names a `[build-system]` holding no exact pin on hatchling, under
        any spelling its normalized name allows, then each build constraint
        naming a range.
        """
        manifest = self.checkout.manifest
        if not any(
            canonicalize_name(requirement.name) == "hatchling"
            and self.exact(requirement)
            for requirement in manifest.requires
        ):
            yield Finding(
                file    = manifest.file,
                message = "The `requires` of `[build-system]` holds no exact pin on "
                "hatchling"
            )

        for constraint in manifest.constraints:
            if not self.exact(constraint):
                yield Finding(
                    file    = manifest.file,
                    message = f"The build constraint `{constraint}` in `[tool.uv]` "
                    "is not an exact pin"
                )


class RunCheck(Check):
    """
    Every `uv run` a task or a program on the path runs opens its arguments
    on `--exact --locked`, so it installs exactly what `uv.lock` resolves
    and stops where the lockfile lags its manifest.
    """

    @property
    def sources(self) -> list[tuple[Path, str]]:
        """
        Pairs each file a task or a program on the path runs from, relative
        to the root, with the lines a shell runs out of it, where a task
        declared in TOML gives the lines it runs in place of a file's.
        """
        root = self.checkout.root
        return [
            (task.source.relative_to(root), "\n".join(task.run)) if task.file is None
            else (task.file.relative_to(root), self.commands(task.file))
            for task in self.checkout.tasks
        ] + [(path, self.commands(root / path)) for path in self.checkout.scripts]

    @staticmethod
    def commands(path: Path) -> str:
        """
        Reads the lines a shell runs out of the file at `path`, which for a
        Python task is the shebang alone and for any other file every line
        but a comment.
        """
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".py":
            return text.partition("\n")[0]

        return sub(r"(?m)^\s*#(?!!).*$", "", text)

    def scan(self) -> Iterator[Finding]:
        """
        Names each file running `uv run` without `--exact --locked` opening
        its arguments, reading a backslash continuation as a space.
        """
        for path, lines in self.sources:
            if search(r"\buv[\s\\]+run\b(?![\s\\]+--exact[\s\\]+--locked\b)", lines):
                yield Finding(
                    file    = path,
                    message = f"`{path}` runs `uv run` without `--exact --locked` "
                    "opening its arguments"
                )


class WheelCheck(Check):
    """
    Every `packages` entry of the wheel holds a module, and every script
    `[project.scripts]` declares targets a module the wheel carries, since
    hatchling skips a `packages` entry naming nothing and still writes the
    wheel.
    """

    @property
    def modules(self) -> dict[Path, set[str]]:
        """
        Maps each `packages` entry of the wheel to the dotted name of every
        module under it, the import name opening on the entry's last folder
        as hatchling installs it.
        """
        root = self.checkout.root
        return {
            entry: {
                ".".join(
                    path.relative_to((root / entry).parent).with_suffix("").parts
                ).removesuffix(".__init__")
                for path in (root / entry).rglob("*.py")
            }
            for entry in self.checkout.manifest.packages
        }

    def scan(self) -> Iterator[Finding]:
        """
        Names each `packages` entry holding no module, then each script
        targeting a module no entry holds.
        """
        manifest = self.checkout.manifest
        modules  = self.modules
        for entry, names in modules.items():
            if not names:
                yield Finding(
                    file    = manifest.file,
                    message = f"The wheel `packages` entry `{entry}` holds no module"
                )

        carried = set().union(*modules.values())
        for script, target in manifest.scripts.items():
            if (module := target.partition(":")[0]) not in carried:
                yield Finding(
                    file    = manifest.file,
                    message = f"The `{script}` script targets `{module}`, which the "
                    "wheel does not carry"
                )
