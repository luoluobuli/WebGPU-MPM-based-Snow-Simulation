import constantsModuleSrc from "./constants.wgsl?raw";
import uniformsModuleSrc from "./uniforms.wgsl?raw";
import randomNumberGeneratorsModuleSrc from "./randomNumberGenerators.wgsl?raw";
import matrixOpsModuleSrc from "./matrixOps.wgsl?raw";
import mpmOpsModuleSrc from "./mpmOps.wgsl?raw";
import stressTensorOpsModuleSrc from "./stressTensorOps.wgsl?raw";
import plasticityOpsModuleSrc from "./plasticityOps.wgsl?raw";
import cameraOpsModuleSrc from "./cameraOps.wgsl?raw";

export const prelude = `\
${constantsModuleSrc}
${uniformsModuleSrc}
${randomNumberGeneratorsModuleSrc}
${matrixOpsModuleSrc}
${mpmOpsModuleSrc}
${stressTensorOpsModuleSrc}
${plasticityOpsModuleSrc}
${cameraOpsModuleSrc}
`;

export const attachPrelude = (moduleSrc: string) => `\
${prelude}
${moduleSrc}`;