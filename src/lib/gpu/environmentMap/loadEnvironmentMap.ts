import environmentMapUrl from "$lib/assets/qwantani_sunset_puresky_2k.png?url";
// import { HDRLoader } from "three/addons/loaders/HDRLoader.js";

export const loadEnvironmentMap = async () => {
    const response = await fetch(environmentMapUrl);
    const blob = await response.blob();
    const imageBitmap = await createImageBitmap(blob);
    return imageBitmap;
};