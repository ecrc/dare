#!/usr/bin/env python3
"""Add wall-clock checkpoints to an isolated production source; no new fences."""
from pathlib import Path
import sys
p=Path(sys.argv[1]); s=p.read_text()
def replace(old,new):
 global s
 assert s.count(old)==1, (old,s.count(old))
 s=s.replace(old,new)
replace('int main(int argc, char *argv[]) {', '''int main(int argc, char *argv[]) {
    const auto wall_start = std::chrono::steady_clock::now();
    auto wall_last = wall_start;
    const auto wall_mark = [&](const char *name) {
        const auto now = std::chrono::steady_clock::now();
        printf("WALL_PHASE %s %.9f\\n", name, std::chrono::duration<double>(now-wall_last).count());
        wall_last = now;
    };''')
replace('    CHAMELEON_Init( NCPUS, NGPUS );','    wall_mark("arguments");\n    CHAMELEON_Init( NCPUS, NGPUS );\n    wall_mark("runtime_init");')
replace('    if (!tilepath.empty()) {','    wall_mark("descriptors_workspaces");\n    if (!tilepath.empty()) {')
replace('    /* Start kernel statistics */','    wall_mark("input");\n    /* Start kernel statistics */')
replace('    START_TIMING( time );','    wall_mark("precompute");\n    START_TIMING( time );')
replace('    STOP_TIMING( time );','    STOP_TIMING( time );\n    wall_mark("solver");')
replace('    /* Stop tracing */','    wall_mark("output");\n    /* Stop tracing */')
replace('    CHAMELEON_Sequence_Destroy( sequence );','    wall_mark("reporting");\n    CHAMELEON_Sequence_Destroy( sequence );')
replace('    /* Finalize CHAMELEON */\n    CHAMELEON_Finalize();','''    wall_mark("descriptor_cleanup");
    /* Finalize CHAMELEON */
    CHAMELEON_Finalize();
    wall_mark("runtime_finalize");
    printf("WALL_MAIN %.9f\\n", std::chrono::duration<double>(std::chrono::steady_clock::now()-wall_start).count());''')
p.write_text(s)
