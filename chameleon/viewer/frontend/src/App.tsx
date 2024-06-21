/*
* Copyright (c) Meta Platforms, Inc. and affiliates.
*
* This source code is licensed under the Chameleon License found in the
* LICENSE file in the root directory of this source tree.
*/
import { Route, Routes } from "react-router-dom";

import { GenerateMixedModal } from "./components/pages/GenerateMixedModal";

import { BasicNavbar, NavContent } from "./components/ri-components/navbars/BasicNavbar";

// JSON Viewer specific css
import "react18-json-view/src/style.css";

function App() {
  const navContent: NavContent = {
    title: "Chameleon",
    description: "Model Input/Output Viewer",
    githubLink: "https://github.com/facebookresearch/chameleon",
    showHomeLink: true,
    navItems: [
      {
        id: "item1",
        url: "/item1",
        title: "Item 1",
        showArrowIcon: true,
      },
      {
        id: "item2",
        url: "/item2",
        title: "Item 2",
        showArrowIcon: false,
      },
      {
        id: "item3",
        title: "Item 3 (No URL)",
        showArrowIcon: false,
      },
    ],
  };

  return (
    <div>
      <BasicNavbar
        content={navContent}
      />

      <Routes>
        <Route index path="*" element={
          <div className="relative lg:px-12 px-5 flex flex-col">
            <GenerateMixedModal />
          </div>
        }
        />
      </Routes>
    </div>
  );
}

export default App;
