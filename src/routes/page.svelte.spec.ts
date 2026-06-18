import { page } from "vitest/browser";
import { afterEach, describe, expect, it } from "vitest";
import { render } from "vitest-browser-svelte";
import Page from "./+page.svelte";

const originalUrl = location.href;

describe("/+page.svelte", () => {
    afterEach(() => {
        history.replaceState(null, "", originalUrl);
    });

    it("should render the simulation viewer shell", async () => {
        render(Page);

        await expect.element(page.getByRole("main")).toBeInTheDocument();
        await expect.element(page.getByRole("heading", { name: "Status" })).toBeInTheDocument();
    });

    it("should show the simulation timeline when `?timeline` is present", async () => {
        history.pushState(null, "", "/?timeline");
        render(Page);

        await expect.element(page.getByRole("heading", { name: "Timeline" })).toBeInTheDocument();
    });

    it("should not show the simulation timeline when `?animated` is present", async () => {
        history.pushState(null, "", "/?animated");
        render(Page);

        await expect.element(page.getByRole("heading", { name: "Timeline" })).not.toBeInTheDocument();
    });
});
