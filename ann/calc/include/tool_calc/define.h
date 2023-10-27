#ifndef INCLUDE_TOOL_CALC_DEFINE_H
#define INCLUDE_TOOL_CALC_DEFINE_H

#define TOOL_CALC_EPSINON (0.000001)

/*
TOOL_CALC_MKL: compile with MKL(Intel Cblas)
TOOL_CALC_ARM: compile with ARM(arm assembler/neon)
TOOL_CALC_OPENMP: compile with openmp, not in mkl

1. TOOL_CALC_MKL: server(intel CPU)
2. TOOL_CALC_ARM: serial arm SDK
3. TOOL_CALC_ARM + TOOL_CALC_OPENMP: parallel arm SDK
4. TOOL_CALC_OPENMP: for cpu test
5. nothing: for cpu test
*/

#endif /* INCLUDE_TOOL_CALC_DEFINE_H */
