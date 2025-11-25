import uniformsModuleSrc from "./uniforms.wgsl?raw";
import randomNumberGeneratorsModuleSrc from "./randomNumberGenerators.wgsl?raw";
import matrixOpsModuleSrc from "./matrixOps.wgsl?raw";
import mpmOpsModuleSrc from "./mpmOps.wgsl?raw";
import stressTensorOpsModuleSrc from "./stressTensorOps.wgsl?raw";
import plasticityOpsModuleSrc from "./plasticityOps.wgsl?raw";

export const prelude = `
${uniformsModuleSrc}
${randomNumberGeneratorsModuleSrc}
${matrixOpsModuleSrc}
${mpmOpsModuleSrc}
${stressTensorOpsModuleSrc}
${plasticityOpsModuleSrc}`;

export const attachPrelude = (moduleSrc: string) => `${prelude}
${moduleSrc}`;