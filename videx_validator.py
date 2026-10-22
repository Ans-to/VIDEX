"""
VIDEX Programming Language — Semantic Validator & Optimizer (Stage 2.5)
=======================================================================

Pipeline Stage 2.5:
    VIDEX Source Code -> Lexer -> Parser -> [VALIDATOR] -> Video Engine -> MP4

Validates timeline invariants on a parsed ``VideoProgram`` AST and applies
VIDEX's core self-healing philosophy ("If something is not specified -> choose
a smart default automatically") while emitting structured compiler diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

from videx_parser import Command, FrameBlock, VideoProgram


@dataclass(frozen=True)
class CompilerDiagnostic:
    """Structured compiler notice/warning produced during AST validation."""

    severity: str  # "INFO" | "WARNING"
    line: int
    code: str
    message: str

    def __str__(self) -> str:
        icon = "⚠️" if self.severity == "WARNING" else "ℹ️"
        return f"{icon} [{self.code}] Line {self.line}: {self.message}"


class VidexValidator:
    """Validates, normalizes, and optimizes a ``VideoProgram`` AST before rendering."""

    OBJECT_MUTATING_ACTIONS: frozenset[str] = frozenset(
        {
            "scale_object",
            "rotate_object",
            "move_object",
            "set_object_opacity",
            "flip_object",
        }
    )

    def validate(self, program: VideoProgram) -> list[CompilerDiagnostic]:
        diagnostics: list[CompilerDiagnostic] = []

        # 1. Ensure even dimensions required by H.264 `yuv420p` chroma subsampling
        if program.width % 2 != 0 or program.height % 2 != 0:
            old_w, old_h = program.width, program.height
            program.width += program.width % 2
            program.height += program.height % 2
            diagnostics.append(
                CompilerDiagnostic(
                    severity="WARNING",
                    line=1,
                    code="VDX-W001",
                    message=(
                        f"Adjusted odd resolution {old_w}x{old_h} -> "
                        f"{program.width}x{program.height} for H.264 yuv420p compatibility."
                    ),
                )
            )

        known_objects: set[str] = set()
        for cmd in program.global_commands:
            if cmd.action == "place_object":
                known_objects.add(str(cmd.params["name"]))

        # 2. Sort `second` blocks chronologically and validate each frame block
        program.seconds.sort(key=lambda s: s.second_number)

        for sec in program.seconds:
            merged_by_index: dict[int, FrameBlock] = {}
            for fb in sec.frames:
                clamped_idx = fb.index
                if fb.index < 1 or fb.index > program.fps:
                    clamped_idx = max(1, min(program.fps, fb.index))
                    diagnostics.append(
                        CompilerDiagnostic(
                            severity="WARNING",
                            line=fb.line,
                            code="VDX-W002",
                            message=(
                                f"Frame index {fb.index} in second {sec.second_number} "
                                f"exceeds 1..{program.fps} at {program.fps} fps; clamped to {clamped_idx}."
                            ),
                        )
                    )

                if clamped_idx in merged_by_index:
                    diagnostics.append(
                        CompilerDiagnostic(
                            severity="INFO",
                            line=fb.line,
                            code="VDX-I001",
                            message=(
                                f"Merged duplicate frame/keyframe {clamped_idx} in second {sec.second_number}."
                            ),
                        )
                    )
                    merged_by_index[clamped_idx].commands.extend(fb.commands)
                else:
                    fb.index = clamped_idx
                    merged_by_index[clamped_idx] = fb

            sec.frames = [merged_by_index[k] for k in sorted(merged_by_index)]

            # 3. Auto-instantiate implicitly referenced objects per VIDEX Core Philosophy
            for fb in sec.frames:
                injected: list[Command] = []
                for cmd in fb.commands:
                    if cmd.action == "set_background":
                        # Scene change resets active stage objects unless re-placed
                        known_objects.clear()
                    elif cmd.action == "place_object":
                        known_objects.add(str(cmd.params["name"]))
                    elif cmd.action == "remove_object":
                        known_objects.discard(str(cmd.params["name"]))
                    elif cmd.action in self.OBJECT_MUTATING_ACTIONS:
                        obj_name = str(cmd.params["name"])
                        if obj_name not in known_objects:
                            known_objects.add(obj_name)
                            injected.append(
                                Command(
                                    action="place_object",
                                    params={"name": obj_name, "position": "center", "file": None},
                                    line=cmd.line,
                                )
                            )
                            diagnostics.append(
                                CompilerDiagnostic(
                                    severity="INFO",
                                    line=cmd.line,
                                    code="VDX-I002",
                                    message=(
                                        f'Object "{obj_name}" was modified before explicit placement; '
                                        f"auto-placed at center."
                                    ),
                                )
                            )
                    injected.append(cmd)
                fb.commands = injected

        return diagnostics
