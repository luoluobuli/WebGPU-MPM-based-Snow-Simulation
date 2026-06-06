import { page } from "vitest/browser";
import { describe, expect, it } from "vitest";
import { render } from "vitest-browser-svelte";
import Page from "./+page.svelte";

describe("/environment/+page.svelte", () => {
    it("should render the shared simulation viewer shell", async () => {
        render(Page);

        await expect.element(page.getByRole("main")).toBeInTheDocument();
        await expect.element(page.getByRole("heading", { name: "Status" })).toBeInTheDocument();
    });
});
